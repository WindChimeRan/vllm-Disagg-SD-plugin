"""Incremental, ragged DFlash1 inference on a dedicated GPU.

Only committed target features enter the persistent context cache. Query K/V
occupies scratch positions immediately after that context and is overwritten
on the next call. Rotary positions remain independent of physical cache slots,
so a future query can cross an unresolved prefix gap.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from disagg_sd.cache import ContextSlots
from disagg_sd.features import FeatureDelta
from disagg_sd.types import DraftJob, DraftResult, RequestKey


@dataclass(frozen=True)
class DraftInput:
    job: DraftJob
    features: FeatureDelta


@dataclass(frozen=True)
class DraftBatch:
    inputs: tuple[DraftInput, ...]
    retired: tuple[RequestKey, ...] = ()


class _MetadataBuffer:
    """One reusable pinned upload; the worker completes before reusing it."""

    def __init__(self, dtype: torch.dtype, device: torch.device):
        self.dtype, self.device = dtype, device
        self.capacity = 0

    def upload(self, values: list[int]) -> torch.Tensor:
        count = len(values)
        if count > self.capacity:
            self.capacity = max(count, 2 * self.capacity)
            self.host = torch.empty(self.capacity, dtype=self.dtype, pin_memory=True)
            self.gpu = torch.empty(self.capacity, dtype=self.dtype, device=self.device)
        self.host[:count].copy_(torch.tensor(values, dtype=self.dtype))
        self.gpu[:count].copy_(self.host[:count], non_blocking=True)
        return self.gpu[:count]


@dataclass(frozen=True)
class _BatchLayout:
    ids: torch.Tensor
    positions: torch.Tensor
    slots: torch.Tensor
    query_offsets: torch.Tensor
    key_lengths: torch.Tensor
    block_table: torch.Tensor
    max_query: int
    max_key: int
    offsets: tuple[int, ...]


def _norm(module: nn.Module, values: torch.Tensor) -> torch.Tensor:
    from vllm._custom_ops import rms_norm

    if values.numel() == 0:
        return values
    output = torch.empty_like(values)
    rms_norm(output, values, module.weight, module.variance_epsilon)
    return output


class DFlashBackend:
    @torch.inference_mode()
    def __init__(
        self,
        path: str,
        device: int,
        embedding: torch.Tensor,
        head: torch.Tensor,
        block_size: int,
        max_num_seqs: int,
        max_model_len: int,
    ):
        from dflash.model import DFlashDraftModel

        self.device = torch.device("cuda", device)
        self.source_device = embedding.device
        self.transfer_stream = torch.cuda.Stream(device=self.source_device)
        self.contexts = ContextSlots(max_num_seqs, max_model_len)
        self.page_size = 256
        self.pages_per_request = (max_model_len + 255) // 256
        self.cache_length = self.pages_per_request * self.page_size
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.Stream(device=self.device)
            self.model = (
                DFlashDraftModel.from_pretrained(
                    path, dtype=embedding.dtype, attn_implementation="sdpa"
                )
                .to(self.device)
                .eval()
            )
            if self.model.block_size != block_size + 1:
                raise ValueError("Use the checkpoint's block size minus its anchor")
            if (
                embedding.shape[1] != self.model.config.hidden_size
                or head.shape != embedding.shape
                or embedding.shape[0] != self.model.config.vocab_size
            ):
                raise ValueError("DFlash and target embedding/head dimensions differ")
            if any(
                layer.self_attn.is_causal
                or layer.self_attn.sliding_window is not None
                or layer.attention_conv is not None
                or layer.mlp_conv is not None
                for layer in self.model.layers
            ):
                raise ValueError("This backend requires non-causal DFlash1 layers")
            self.embedding = embedding.detach().to(self.device, copy=True)
            self.head = nn.Linear(
                head.shape[1],
                head.shape[0],
                bias=False,
                device=self.device,
                dtype=head.dtype,
            )
            self.head.weight.copy_(head)
            self.head.requires_grad_(False)
            cfg = self.model.config
            shape = (
                max_num_seqs * self.pages_per_request,
                self.page_size,
                cfg.num_key_value_heads,
                cfg.head_dim,
            )
            self.kv = [
                (
                    torch.empty(shape, device=self.device, dtype=embedding.dtype),
                    torch.empty(shape, device=self.device, dtype=embedding.dtype),
                )
                for _ in self.model.layers
            ]
            self.scale = torch.ones((), device=self.device, dtype=torch.float32)
            self._long = _MetadataBuffer(torch.int64, self.device)
            self._int = _MetadataBuffer(torch.int32, self.device)
            torch.cuda.synchronize(self.device)

    def _layout(self, batch: tuple[DraftInput, ...]) -> _BatchLayout:
        context_positions, context_slots = [], []
        query_positions, query_slots, ids = [], [], []
        offsets, lengths, blocks = [0], [], []
        for item in batch:
            job, delta = item.job, item.features
            if delta.end != job.anchor.position:
                raise ValueError("Feature delta does not reach the draft anchor")
            slot = self.contexts.extend(job.key, delta.start, delta.end)
            count = job.count + 1
            if delta.end + count > self.cache_length:
                raise ValueError("Draft queries exceed context cache capacity")
            base = slot * self.cache_length
            context_positions.extend(range(delta.start, delta.end))
            context_slots.extend(range(base + delta.start, base + delta.end))
            query_positions.extend(
                [job.anchor.position, *range(job.start, job.start + job.count)]
            )
            query_slots.extend(range(base + delta.end, base + delta.end + count))
            ids.extend([job.anchor.token] + [self.model.mask_token_id] * job.count)
            offsets.append(offsets[-1] + count)
            lengths.append(delta.end + count)
            blocks.extend(
                range(
                    slot * self.pages_per_request, (slot + 1) * self.pages_per_request
                )
            )
        nquery = len(ids)
        positions = context_positions + query_positions
        packed = self._long.upload(ids + positions + context_slots + query_slots)
        integers = self._int.upload(offsets + lengths + blocks)
        size = len(batch)
        return _BatchLayout(
            packed[:nquery],
            packed[nquery : nquery + len(positions)],
            packed[nquery + len(positions) :],
            integers[: size + 1],
            integers[size + 1 : 2 * size + 1],
            integers[2 * size + 1 :].view(size, self.pages_per_request),
            max(x.job.count + 1 for x in batch),
            max(lengths),
            tuple(offsets),
        )

    def _transfer(self, batch: tuple[DraftInput, ...]) -> torch.Tensor:
        chunks = [chunk for item in batch for chunk in item.features.chunks]
        expected = sum(x.features.end - x.features.start for x in batch)
        if sum(chunk.values.shape[0] for chunk in chunks) != expected:
            raise ValueError("Feature delta storage does not match its positions")
        if not chunks:
            return self.embedding.new_empty((0, self.model.fc.in_features))
        # Source copies use their own stream, avoiding a dependency on future
        # target forwards. The cross-device copy orders this stream and ours.
        with torch.cuda.stream(self.transfer_stream):
            for chunk in chunks:
                self.transfer_stream.wait_event(chunk.ready)
            packed = torch.cat([chunk.values for chunk in chunks], dim=0)
            return packed.to(self.device, non_blocking=True)

    def _forward(self, features: torch.Tensor, layout: _BatchLayout) -> torch.Tensor:
        from dflash.model import apply_rotary_pos_emb
        from vllm._custom_ops import reshape_and_cache_flash
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        hidden = F.embedding(layout.ids, self.embedding)
        context = _norm(self.model.hidden_norm, self.model.fc(features))
        cos, sin = self.model.rotary_emb(hidden[None], layout.positions[None])
        # Reuse the checkpoint's projections, norms and MLPs. Only attention
        # storage/execution differs from upstream's dense SDPA forward.
        for layer, (keys, values) in zip(self.model.layers, self.kv, strict=True):
            attn = layer.self_attn
            residual = hidden
            noise = _norm(layer.input_layernorm, hidden)
            q = _norm(
                attn.q_norm,
                attn.q_proj(noise).view(
                    -1, attn.config.num_attention_heads, attn.head_dim
                ),
            )
            k = torch.cat((attn.k_proj(context), attn.k_proj(noise))).view(
                -1, attn.config.num_key_value_heads, attn.head_dim
            )
            v = torch.cat((attn.v_proj(context), attn.v_proj(noise))).view_as(k)
            k = _norm(attn.k_norm, k)
            q, k = apply_rotary_pos_emb(
                q.transpose(0, 1)[None], k.transpose(0, 1)[None], cos, sin
            )
            q, k = q[0].transpose(0, 1), k[0].transpose(0, 1)
            reshape_and_cache_flash(
                k, v, keys, values, layout.slots, "auto", self.scale, self.scale
            )
            output = flash_attn_varlen_func(
                q,
                keys,
                values,
                layout.max_query,
                layout.query_offsets,
                layout.max_key,
                seqused_k=layout.key_lengths,
                block_table=layout.block_table,
                softmax_scale=attn.scaling,
                causal=False,
                fa_version=2,
                num_splits=1,
            )
            hidden = residual + attn.o_proj(output.flatten(1))
            hidden = hidden + layer.mlp(_norm(layer.post_attention_layernorm, hidden))
        hidden = _norm(self.model.norm, hidden)
        return self.model.compute_logits(hidden, self.head).argmax(-1)

    @torch.inference_mode()
    def __call__(self, batch: DraftBatch) -> tuple[DraftResult, ...]:
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            for key in batch.retired:
                self.contexts.remove(key)
            if not batch.inputs:
                return ()
            with torch.profiler.record_function("disagg.draft"):
                layout = self._layout(batch.inputs)
                features = self._transfer(batch.inputs)
                tokens = self._forward(features, layout).cpu().tolist()
        return tuple(
            DraftResult(item.job, tuple(tokens[start + 1 : end]))
            for item, start, end in zip(
                batch.inputs, layout.offsets[:-1], layout.offsets[1:], strict=True
            )
        )

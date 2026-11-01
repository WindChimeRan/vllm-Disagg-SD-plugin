"""Opt-in numerical check against the upstream dense DFlash1 forward.

Run with DISAGG_TEST_CUDA=1 and two visible GPUs. No downloaded models needed.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("DISAGG_TEST_CUDA") != "1", reason="two-GPU integration check"
)


def test_cached_ragged_forward_matches_full_context(tmp_path):
    import torch
    from dflash.model import DFlashDraftModel
    from transformers import Qwen3Config

    from disagg_sd.dflash import DFlashBackend, DraftBatch, DraftInput
    from disagg_sd.dflash_worker import DFlashWorker
    from disagg_sd.features import FeatureChunk, FeatureDelta
    from disagg_sd.types import Anchor, DraftJob, RequestKey

    torch.manual_seed(17)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_target_layers=4,
        block_size=16,
        initializer_range=0.1,
        dflash_config={"mask_token_id": 127, "target_layer_ids": [1, 3]},
    )
    DFlashDraftModel(config).save_pretrained(tmp_path)
    embedding = torch.randn(128, 64, device="cuda:0", dtype=torch.float16)
    backend = DFlashBackend(str(tmp_path), 1, embedding, embedding, 15, 2, 512)
    worker = DFlashWorker(str(tmp_path), 1, embedding, embedding, 15, 2, 512, 60)
    full = {}
    lengths = {}
    captured = []
    hook = backend.head.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )

    def step(specs, retired=()):
        inputs = []
        for key, added, gap, count in specs:
            start = lengths.get(key, 0)
            features = torch.randn(added, 128, device="cuda:0", dtype=torch.float16)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(0))
            full[key] = torch.cat((full.get(key, features[:0]), features))
            lengths[key] = end = start + added
            job = DraftJob(len(inputs), key, Anchor(end, 13), end + 1 + gap, count)
            inputs.append(
                DraftInput(
                    job, FeatureDelta(start, end, (FeatureChunk(features, ready),))
                )
            )
        batch = DraftBatch(tuple(inputs), retired)
        expected_tokens = backend(batch)
        worker.submit(batch)
        assert worker.receive(wait=True) == expected_tokens
        actual = captured.pop()
        expected = []
        with torch.inference_mode(), torch.cuda.device(1):
            for item in inputs:
                job = item.job
                ids = torch.tensor([[13] + [127] * job.count], device="cuda:1")
                positions = torch.tensor(
                    [
                        [
                            *range(job.anchor.position),
                            job.anchor.position,
                            *range(job.start, job.start + job.count),
                        ]
                    ],
                    device="cuda:1",
                )
                expected.append(
                    backend.model(
                        target_hidden=full[job.key].to("cuda:1")[None],
                        noise_embedding=torch.nn.functional.embedding(
                            ids, backend.embedding
                        ),
                        position_ids=positions,
                        use_cache=False,
                    )[0]
                )
                # Check the committed cache directly: residual connections can
                # otherwise hide a broken attention path in a tiny random model.
                from dflash.model import apply_rotary_pos_emb

                length = job.anchor.position
                slot = backend.contexts.extend(job.key, length, length)
                context = backend.model.hidden_norm(
                    backend.model.fc(full[job.key].to("cuda:1"))
                )
                cos, sin = backend.model.rotary_emb(
                    context[None], torch.arange(length, device="cuda:1")[None]
                )
                for layer, (keys, values) in zip(
                    backend.model.layers, backend.kv, strict=True
                ):
                    attention = layer.self_attn
                    k = attention.k_norm(attention.k_proj(context).view(length, 2, 16))
                    v = attention.v_proj(context).view(length, 2, 16)
                    k = k.transpose(0, 1)[None]
                    _, k = apply_rotary_pos_emb(k, k, cos, sin)
                    cached_k = keys.view(2, backend.cache_length, 2, 16)[slot, :length]
                    cached_v = values.view(2, backend.cache_length, 2, 16)[
                        slot, :length
                    ]
                    torch.testing.assert_close(
                        cached_k, k[0].transpose(0, 1), atol=0.025, rtol=0.025
                    )
                    torch.testing.assert_close(cached_v, v, atol=0.025, rtol=0.025)
            torch.testing.assert_close(
                actual, torch.cat(expected), atol=0.025, rtol=0.025
            )

    a, b, reused = RequestKey("a", 0), RequestKey("b", 1), RequestKey("a", 2)
    try:
        step([(a, 19, 0, 15), (b, 273, 16, 7)])
        step([(b, 3, 0, 11), (a, 2, 16, 15)])
        step([(a, 0, 8, 4)])
        step([(reused, 7, 0, 15), (b, 1, 5, 3)], retired=(a,))
    finally:
        hook.remove()
        worker.close()

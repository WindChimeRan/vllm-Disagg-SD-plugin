import torch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator
from vllm.v1.worker.gpu_worker import Worker

from disagg_sd.dflash_worker import DFlashWorker
from disagg_sd.vllm.session import FeatureSpan, TargetFeatures, get_session


class FeatureCollector(BaseSpeculator):
    """Collect target outputs; proposals arrive through the scheduler adapter."""

    def __init__(self, vllm_config, device):
        self.session = get_session(vllm_config)
        self.device = device
        self.width = vllm_config.speculative_config.num_speculative_tokens

    def init_cudagraph_manager(self, cudagraph_mode) -> None:
        pass  # Eager mode is checked when the session is constructed.

    def capture(self) -> None:
        raise RuntimeError("Disaggregated runtime does not support CUDA graphs yet")

    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        dp_sync_state=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
        num_speculative_tokens=None,
    ):
        if self.session.serving and not dummy_run and not is_profile:
            if self.session.capture is not None:
                raise RuntimeError(
                    "A target batch was not consumed before the next one"
                )
            if aux_hidden_states is None:
                raise RuntimeError("DFlash needs target auxiliary hidden states")
            self.session.capture = TargetFeatures(
                tuple(
                    FeatureSpan(
                        rid,
                        int(input_batch.num_computed_tokens_np[i]),
                        int(input_batch.query_start_loc_np[i]),
                        int(input_batch.num_scheduled_tokens[i]),
                    )
                    for i, rid in enumerate(input_batch.req_ids)
                ),
                aux_hidden_states,
            )
        return torch.full(
            (input_batch.num_reqs, self.width),
            -1,
            dtype=torch.int64,
            device=self.device,
        )


class DisaggModelRunner(GPUModelRunner):
    def load_model(self, *args, **kwargs) -> None:
        super().load_model(*args, **kwargs)
        session = get_session(self.vllm_config)
        if session.options.draft_device == self.device.index:
            raise ValueError("Target and drafter must use different CUDA devices")
        vocab = self.vocab_size
        session.worker = DFlashWorker(
            self.speculative_config.model,
            session.options.draft_device,
            self.model.model.embed_tokens.weight[:vocab],
            self.model.lm_head.weight[:vocab],
            self.speculative_config.num_speculative_tokens,
            self.scheduler_config.max_num_seqs,
            self.model_config.max_model_len,
            session.options.draft_timeout,
        )
        shape = (
            self.scheduler_config.max_num_seqs,
            self.speculative_config.num_speculative_tokens + 1,
        )
        self._draft_host = torch.empty(shape, dtype=torch.int64, pin_memory=True)
        self._draft_gpu = torch.empty(shape, dtype=torch.int64, device=self.device)

    def prepare_inputs(
        self, scheduler_output, batch_req_state, batch_desc, num_active_loras
    ):
        candidates = scheduler_output.scheduled_spec_decode_tokens
        if candidates:
            count = len(candidates)
            host = self._draft_host[:count]
            host.fill_(-1)
            rows = host.numpy()
            for row, (rid, tokens) in zip(rows, candidates.items(), strict=True):
                row[0] = self.req_states.req_id_to_index[rid]
                row[1 : len(tokens) + 1] = tokens
            gpu = self._draft_gpu[:count]
            gpu.copy_(host, non_blocking=True)
            self.req_states.draft_tokens.index_copy_(0, gpu[:, 0], gpu[:, 1:])
        return super().prepare_inputs(
            scheduler_output, batch_req_state, batch_desc, num_active_loras
        )

    def take_draft_token_ids(self):
        return None


class DisaggWorker(Worker):
    def _make_model_runner(self):
        return DisaggModelRunner(self.vllm_config, self.device)

    def shutdown(self):
        # Drain copies before the native worker releases target device storage.
        session = getattr(self.vllm_config, "_disagg_sd_session", None)
        if session is not None:
            session.close()
        super().shutdown()

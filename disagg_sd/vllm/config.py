from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig


class DisaggSpeculatorConfig(DFlashSpeculatorConfig):
    speculator_class = "disagg_sd.vllm.worker.FeatureCollector"
    supported_options = frozenset({"speculator_options"})
    supports_watermarking = False

    def max_num_draft_seqs(self, max_num_batched_tokens: int) -> int | None:
        # The draft batch has its own GPU and allocation budget.
        return None

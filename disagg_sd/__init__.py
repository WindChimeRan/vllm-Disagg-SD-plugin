"""The core runtime is independent of torch and vLLM."""


def register() -> None:
    from vllm.v1.worker.gpu.spec_decode.speculator_config import (
        register_speculative_method,
    )

    register_speculative_method(
        "disagg_dflash", "disagg_sd.vllm.config.DisaggSpeculatorConfig"
    )

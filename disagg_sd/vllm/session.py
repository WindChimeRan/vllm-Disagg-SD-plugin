"""Engine-local binding between the scheduler, target, and draft transport."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from disagg_sd.config import RuntimeConfig
from disagg_sd.features import FeatureStore
from disagg_sd.policy import make_policy
from disagg_sd.runtime import Coordinator
from disagg_sd.transport import Worker
from disagg_sd.types import DraftJob, DraftResult, RequestKey

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from disagg_sd.dflash import DraftBatch


@dataclass(frozen=True)
class FeatureSpan:
    request_id: str
    start: int
    offset: int
    count: int


@dataclass(frozen=True)
class TargetFeatures:
    spans: tuple[FeatureSpan, ...]
    layers: list[torch.Tensor]


class Session:
    def __init__(self, config: "VllmConfig"):
        spec = config.speculative_config
        if spec is None or spec.custom_method != "disagg_dflash":
            raise ValueError("Disagg workers require method='disagg_dflash'")
        self.options = RuntimeConfig.parse(spec.speculator_options or {})
        pc = config.parallel_config
        if (
            pc.tensor_parallel_size,
            pc.pipeline_parallel_size,
            pc.data_parallel_size,
        ) != (1, 1, 1):
            raise ValueError("The local prototype requires TP=PP=DP=1")
        if pc.distributed_executor_backend != "uni":
            raise ValueError("The local prototype requires executor='uni'")
        if config.scheduler_config.async_scheduling:
            raise ValueError(
                "Set async_scheduling=False: one target batch may be in flight"
            )
        if not config.use_v2_model_runner or not config.model_config.enforce_eager:
            raise ValueError(
                "The prototype requires the V2 runner and enforce_eager=True"
            )
        if config.cache_config.enable_prefix_caching:
            raise ValueError(
                "Prefix caching lacks the target features needed by DFlash"
            )
        if (
            config.lora_config
            or config.kv_transfer_config
            or config.watermark_config
            or config.model_config.is_multimodal_model
            or config.model_config.quantization
        ):
            raise ValueError("Use unquantized text models without LoRA or KV transfer")
        if spec.enable_adaptive_verification or spec.draft_sample_method != "greedy":
            raise ValueError("Use fixed verification with greedy drafting")
        if config.model_config.hf_config.model_type != "qwen3":
            raise ValueError("The initial backend supports Qwen3 + DFlash1")
        self.coordinator = Coordinator(
            make_policy(self.options.policy),
            spec.num_speculative_tokens,
            self.options.max_ahead,
        )
        self.features = FeatureStore()
        self.worker: Worker[DraftBatch, tuple[DraftResult, ...]] | None = None
        self._retired: set[RequestKey] = set()
        self.capture: TargetFeatures | None = None
        self.serving = False
        self._trace = None
        if self.options.trace_path:
            path = Path(self.options.trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._trace = path.open("w")

    def log(self, event: str, **fields) -> None:
        if self._trace is not None:
            self._trace.write(
                json.dumps({"event": event, "time": time.perf_counter(), **fields})
                + "\n"
            )
            self._trace.flush()

    def receive(self, *, wait: bool = False) -> None:
        if self.worker is None:
            raise RuntimeError(
                "Target and scheduler did not share a local worker session"
            )
        results = self.worker.receive(wait=wait)
        if results is not None:
            for result in results:
                used = self.coordinator.complete_draft(result)
                self.log(
                    "draft_complete",
                    request=result.job.key.request_id,
                    job=result.job.job_id,
                    used=used,
                )

    def submit(self, jobs: tuple[DraftJob, ...]) -> None:
        from disagg_sd.dflash import DraftBatch, DraftInput

        if not jobs:
            return
        if self.worker is None:
            raise RuntimeError("Draft worker is not initialized")
        inputs = tuple(
            DraftInput(job, self.features.take(job.key, job.anchor.position))
            for job in jobs
        )
        self.worker.submit(DraftBatch(inputs, tuple(self._retired)))
        self._retired.clear()
        self.log(
            "draft_submit",
            jobs=[
                {
                    "request": j.key.request_id,
                    "job": j.job_id,
                    "start": j.start,
                    "count": j.count,
                    "gap": j.gap,
                }
                for j in jobs
            ],
        )

    def remove(self, request_id: str) -> None:
        if request_id in self.coordinator.request_ids:
            self.retire(self.coordinator.key(request_id))
            self.coordinator.remove(request_id)
            self.log("remove", request=request_id)

    def retire(self, key: RequestKey) -> None:
        self.features.remove(key)
        self._retired.add(key)

    def close(self) -> None:
        self.serving = False
        if self.worker is not None:
            self.worker.close()
            self.worker = None
        for rid in self.coordinator.request_ids:
            self.remove(rid)
        self.capture = None
        self._retired.clear()
        if self._trace is not None:
            self._trace.close()
            self._trace = None


def get_session(config: "VllmConfig") -> Session:
    # UniProcExecutor passes this same config to its worker and scheduler.
    # A per-config binding avoids process-global request state and cross-engine leaks.
    session = getattr(config, "_disagg_sd_session", None)
    if session is None:
        session = Session(config)
        config._disagg_sd_session = session
    return session

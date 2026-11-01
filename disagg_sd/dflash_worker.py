"""CUDA IPC for DFlash batches; request state stays in the target process."""

from dataclasses import dataclass
from functools import partial

import torch

from disagg_sd.dflash import DFlashBackend, DraftBatch, DraftInput
from disagg_sd.features import FeatureChunk, FeatureDelta
from disagg_sd.transport import ProcessWorker
from disagg_sd.types import DraftJob, DraftResult, RequestKey


@dataclass(frozen=True)
class _Input:
    job: DraftJob
    start: int
    end: int


@dataclass(frozen=True)
class _Batch:
    inputs: tuple[_Input, ...]
    retired: tuple[RequestKey, ...]
    values: torch.Tensor | None
    ready: bytes | None


class _Executor:
    def __init__(self, *args):
        torch.set_num_threads(1)
        self.backend = DFlashBackend(*args)

    def __call__(self, batch: _Batch) -> tuple[DraftResult, ...]:
        ready = (
            torch.cuda.Event.from_ipc_handle(batch.values.device, batch.ready)
            if batch.values is not None
            else None
        )
        inputs, offset = [], 0
        for item in batch.inputs:
            count = item.end - item.start
            chunks = ()
            if count:
                chunks = (FeatureChunk(batch.values[offset : offset + count], ready),)
            inputs.append(
                DraftInput(item.job, FeatureDelta(item.start, item.end, chunks))
            )
            offset += count
        # DFlashBackend synchronizes its batched CPU token result before return.
        # No imported feature storage survives this call.
        return self.backend(DraftBatch(tuple(inputs), batch.retired))


class DFlashWorker:
    """Pack one immutable feature buffer and retain it until the child completes."""

    def __init__(
        self,
        path: str,
        device: int,
        embedding: torch.Tensor,
        head: torch.Tensor,
        block_size: int,
        max_num_seqs: int,
        max_model_len: int,
        timeout: float,
    ):
        self._stream = torch.cuda.Stream(device=embedding.device)
        self._pending: tuple[DraftBatch, torch.cuda.Event | None] | None = None
        self._worker = ProcessWorker(
            partial(
                _Executor,
                path,
                device,
                embedding.detach(),
                head.detach(),
                block_size,
                max_num_seqs,
                max_model_len,
            ),
            timeout,
        )

    @property
    def busy(self) -> bool:
        return self._worker.busy

    def submit(self, batch: DraftBatch) -> None:
        if self.busy:
            raise RuntimeError("Only one draft batch may be in flight")
        chunks = [chunk for item in batch.inputs for chunk in item.features.chunks]
        values = ready = None
        with torch.cuda.stream(self._stream):
            if chunks:
                for chunk in chunks:
                    self._stream.wait_event(chunk.ready)
                values = torch.cat([chunk.values for chunk in chunks], dim=0)
                ready = torch.cuda.Event(interprocess=True)
                ready.record(self._stream)
        # Retain source chunks as well as the packed IPC allocation: their
        # producing stream differs from the packing stream.
        self._pending = (batch, ready)
        self._worker.submit(
            _Batch(
                tuple(
                    _Input(x.job, x.features.start, x.features.end)
                    for x in batch.inputs
                ),
                batch.retired,
                values,
                ready.ipc_handle() if ready else None,
            )
        )

    def receive(self, *, wait: bool = False) -> tuple[DraftResult, ...] | None:
        result = self._worker.receive(wait=wait)
        if result is not None:
            self._pending = None
        return result

    def close(self) -> None:
        try:
            self._worker.close()
        finally:
            self._pending = None

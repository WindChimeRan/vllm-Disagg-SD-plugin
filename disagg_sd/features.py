"""Committed feature deltas awaiting transfer to the draft worker."""

from dataclasses import dataclass

import torch

from disagg_sd.types import RequestKey


@dataclass(frozen=True)
class FeatureChunk:
    values: torch.Tensor
    ready: torch.cuda.Event


@dataclass(frozen=True)
class FeatureDelta:
    start: int
    end: int
    chunks: tuple[FeatureChunk, ...]


class FeatureStore:
    def __init__(self) -> None:
        self._chunks: dict[RequestKey, list[FeatureChunk]] = {}
        self._lengths: dict[RequestKey, int] = {}
        self._sent: dict[RequestKey, int] = {}

    def length(self, key: RequestKey) -> int:
        return self._lengths.get(key, 0)

    def append(self, key: RequestKey, start: int, layers: list[torch.Tensor]) -> None:
        if start != self.length(key):
            raise ValueError(
                "Target features must extend a contiguous committed prefix"
            )
        if not layers or layers[0].shape[0] == 0:
            return
        # cat owns its storage: later target forwards can reuse their buffers.
        values = torch.cat(layers, dim=-1)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(values.device))
        self._chunks.setdefault(key, []).append(FeatureChunk(values, ready))
        self._lengths[key] = start + values.shape[0]

    def take(self, key: RequestKey, length: int) -> FeatureDelta:
        if length != self.length(key):
            raise ValueError("Draft anchor does not match available target features")
        delta = FeatureDelta(
            self._sent.get(key, 0), length, tuple(self._chunks.pop(key, ()))
        )
        self._sent[key] = length
        return delta

    def remove(self, key: RequestKey) -> None:
        self._chunks.pop(key, None)
        self._lengths.pop(key, None)
        self._sent.pop(key, None)

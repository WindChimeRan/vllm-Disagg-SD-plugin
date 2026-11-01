"""Position-based work descriptors; no engine objects or device dependencies."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestKey:
    request_id: str
    generation: int


@dataclass(frozen=True)
class Anchor:
    position: int
    token: int


@dataclass(frozen=True)
class DraftJob:
    job_id: int
    key: RequestKey
    anchor: Anchor
    start: int
    count: int

    @property
    def gap(self) -> int:
        return self.start - self.anchor.position - 1


@dataclass(frozen=True)
class DraftResult:
    job: DraftJob
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class RequestView:
    key: RequestKey
    anchor: Anchor
    next_position: int
    limit: int
    candidates: tuple[int, ...]
    buffered_end: int
    pending: DraftJob | None


@dataclass(frozen=True)
class Plan:
    verify: dict[str, tuple[int, ...]]
    drafts: tuple[DraftJob, ...]

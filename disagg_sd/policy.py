"""Scheduling decisions only. Policies never execute models or own KV state."""

from collections.abc import Sequence
from typing import Protocol

from disagg_sd.types import RequestKey, RequestView


class Policy(Protocol):
    retain_suffix: bool

    def admit(self, key: RequestKey) -> None: ...

    def remove(self, key: RequestKey) -> None: ...

    def verifiers(self, requests: Sequence[RequestView]) -> set[RequestKey]: ...

    def draft_start(
        self, request: RequestView, verifying: set[RequestKey]
    ) -> int | None: ...

    def verified(self, keys: set[RequestKey]) -> None: ...


class MineDraftPolicy:
    """Alternate cohorts, balancing admitted requests as in mixed-prefill MineDraft."""

    retain_suffix = False

    def __init__(self) -> None:
        self._groups: dict[RequestKey, int] = {}
        self._turn = 0

    def admit(self, key: RequestKey) -> None:
        sizes = [sum(g == i for g in self._groups.values()) for i in (0, 1)]
        self._groups[key] = 0 if sizes[0] <= sizes[1] else 1

    def remove(self, key: RequestKey) -> None:
        del self._groups[key]
        if not self._groups:
            self._turn = 0

    def verifiers(self, requests: Sequence[RequestView]) -> set[RequestKey]:
        # An empty cohort must not strand the remaining requests.
        if requests and not any(self._groups[r.key] == self._turn for r in requests):
            self._turn = 1 - self._turn
        return {
            r.key
            for r in requests
            if self._groups[r.key] == self._turn and r.candidates
        }

    def draft_start(
        self, request: RequestView, verifying: set[RequestKey]
    ) -> int | None:
        if request.pending or request.candidates or request.key in verifying:
            return None
        draft_group = 1 - self._turn if verifying else self._turn
        if self._groups[request.key] == draft_group:
            return request.next_position
        return None

    def verified(self, keys: set[RequestKey]) -> None:
        # Finished requests may already have left the cohort membership map.
        if keys and self._groups:
            self._turn = 1 - self._turn


class SynchronousPolicy:
    """Control: finish drafting a decode batch before verifying that batch."""

    retain_suffix = False

    def admit(self, key: RequestKey) -> None:
        pass

    def remove(self, key: RequestKey) -> None:
        pass

    def verifiers(self, requests: Sequence[RequestView]) -> set[RequestKey]:
        return {r.key for r in requests if r.candidates}

    def draft_start(
        self, request: RequestView, verifying: set[RequestKey]
    ) -> int | None:
        if verifying or request.pending or request.candidates:
            return None
        return request.next_position

    def verified(self, keys: set[RequestKey]) -> None:
        pass


class OverlapPolicy:
    """Draft beyond an active verify window and retain uncommitted suffix tokens."""

    retain_suffix = True

    def admit(self, key: RequestKey) -> None:
        pass

    def remove(self, key: RequestKey) -> None:
        pass

    def verifiers(self, requests: Sequence[RequestView]) -> set[RequestKey]:
        return {
            r.key for r in requests if r.candidates or r.buffered_end > r.next_position
        }

    def draft_start(
        self, request: RequestView, verifying: set[RequestKey]
    ) -> int | None:
        if request.pending:
            return None
        if not request.candidates:
            return (
                request.next_position
                if request.buffered_end <= request.next_position
                else None
            )
        if request.key in verifying:
            end = request.next_position + len(request.candidates)
            # Native verification also emits a correction/bonus token. Leave
            # that position to the target, including after suffix reverification.
            if request.buffered_end <= end:
                return end + 1
        return None

    def verified(self, keys: set[RequestKey]) -> None:
        pass


def make_policy(name: str) -> Policy:
    if name == "minedraft":
        return MineDraftPolicy()
    if name == "overlap":
        return OverlapPolicy()
    if name == "sync":
        return SynchronousPolicy()
    raise ValueError(f"Unknown policy: {name!r}")

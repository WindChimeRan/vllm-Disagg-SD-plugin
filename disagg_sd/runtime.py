"""Single-owner coordinator for batched draft work and untrusted candidates."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import count

from disagg_sd.policy import Policy
from disagg_sd.types import Anchor, DraftJob, DraftResult, Plan, RequestKey, RequestView


@dataclass
class _Request:
    key: RequestKey
    limit: int
    admitted: bool = False
    next_position: int = 0
    feature_count: int = 0
    anchor: Anchor | None = None
    pending: DraftJob | None = None
    verifying: tuple[int, ...] | None = None
    candidates: dict[int, int] = field(default_factory=dict)

    def view(self, block_size: int) -> RequestView | None:
        if self.anchor is None or self.feature_count != self.anchor.position:
            return None
        tokens = []
        for position in range(self.next_position, self.next_position + block_size):
            if position not in self.candidates:
                break
            tokens.append(self.candidates[position])
        return RequestView(
            self.key,
            self.anchor,
            self.next_position,
            self.limit,
            tuple(tokens),
            max(self.candidates, default=self.next_position - 1) + 1,
            self.pending,
        )


class Coordinator:
    """The engine reports committed progress; this class never commits tokens."""

    def __init__(self, policy: Policy, block_size: int, max_ahead: int) -> None:
        if block_size < 1 or max_ahead < 2 * block_size + 1:
            raise ValueError("max_ahead must hold two draft windows and one bonus")
        self.policy = policy
        self.block_size = block_size
        self.max_ahead = max_ahead
        self._requests: dict[str, _Request] = {}
        self._generations = count()
        self._jobs = count()
        self._verifications: dict[str, RequestKey] = {}

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(self._requests)

    def add(self, request_id: str, limit: int) -> RequestKey:
        if request_id in self._requests:
            raise ValueError(f"Request already registered: {request_id}")
        key = RequestKey(request_id, next(self._generations))
        self._requests[request_id] = _Request(key, limit)
        return key

    def admit(self, request_id: str) -> bool:
        """Enroll only after native scheduling admits this request generation."""
        state = self._requests[request_id]
        if state.admitted:
            return False
        self.policy.admit(state.key)
        state.admitted = True
        return True

    def key(self, request_id: str) -> RequestKey:
        return self._requests[request_id].key

    def ready(self, request_id: str) -> bool:
        return self._requests[request_id].view(self.block_size) is not None

    def remove(self, request_id: str) -> None:
        state = self._requests.pop(request_id, None)
        if state is not None and state.admitted:
            self.policy.remove(state.key)

    def reset(self, request_id: str) -> RequestKey:
        limit = self._requests[request_id].limit
        self.remove(request_id)
        return self.add(request_id, limit)

    def observe(
        self,
        request_id: str,
        committed_length: int,
        last_token: int | None,
        feature_count: int,
    ) -> None:
        state = self._requests[request_id]
        if not state.admitted:
            raise ValueError("Admit the request before reporting target progress")
        if committed_length < state.next_position:
            raise ValueError("Committed progress cannot move backwards; reset first")
        if not 0 <= feature_count <= committed_length:
            raise ValueError("Features cannot extend beyond the committed prefix")
        state.next_position = committed_length
        state.feature_count = feature_count
        state.anchor = (
            Anchor(committed_length - 1, last_token) if last_token is not None else None
        )
        if state.verifying is not None and not self.policy.retain_suffix:
            state.candidates.clear()
        else:
            state.candidates = {
                p: t for p, t in state.candidates.items() if p >= committed_length
            }

    def plan(self, active_ids: Iterable[str], *, can_draft: bool = True) -> Plan:
        views = [
            view
            for rid in active_ids
            if (state := self._requests[rid]).admitted
            and state.next_position < state.limit
            and state.verifying is None
            and (view := state.view(self.block_size)) is not None
        ]
        verifying = self.policy.verifiers(views)
        jobs = []
        if can_draft:
            for view in views:
                start = self.policy.draft_start(view, verifying)
                if start is None or start >= view.limit:
                    continue
                count_ = min(
                    self.block_size,
                    view.limit - start,
                    view.next_position + self.max_ahead - start,
                )
                if start < view.next_position or count_ <= 0:
                    raise ValueError("Policy requested an invalid draft window")
                job = DraftJob(next(self._jobs), view.key, view.anchor, start, count_)
                self._requests[view.key.request_id].pending = job
                jobs.append(job)
        return Plan(
            {v.key.request_id: v.candidates for v in views if v.key in verifying},
            tuple(jobs),
        )

    def complete_draft(self, result: DraftResult) -> bool:
        job = result.job
        state = self._requests.get(job.key.request_id)
        if state is None or state.key != job.key:
            return False
        if state.pending != job:
            raise ValueError("Unexpected or duplicate draft completion")
        if len(result.tokens) != job.count or any(t < 0 for t in result.tokens):
            raise ValueError("Malformed draft result")
        end = min(state.limit, state.next_position + self.max_ahead)
        for position, token in enumerate(result.tokens, job.start):
            if state.next_position <= position < end:
                state.candidates[position] = token
        state.pending = None
        return True

    def begin_verify(self, request_id: str, tokens: tuple[int, ...]) -> None:
        state = self._requests[request_id]
        view = state.view(self.block_size)
        if state.verifying is not None or view is None:
            raise ValueError("Request is not ready for verification")
        if tokens != view.candidates[: len(tokens)]:
            raise ValueError("Verification must extend the prefix contiguously")
        state.verifying = tokens
        self._verifications[request_id] = state.key

    def end_verify(self, request_ids: Iterable[str]) -> None:
        verified = set()
        for rid in request_ids:
            key = self._verifications.pop(rid, None)
            if key is None:
                continue
            verified.add(key)
            state = self._requests.get(rid)
            if state is not None and state.key == key:
                state.verifying = None
        self.policy.verified(verified)

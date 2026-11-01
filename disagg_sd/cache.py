"""Worker-owned identities and committed lengths for a bounded context cache."""

from disagg_sd.types import RequestKey


class ContextSlots:
    def __init__(self, capacity: int, max_length: int):
        self.max_length = max_length
        self._free = list(reversed(range(capacity)))
        self._states: dict[RequestKey, tuple[int, int]] = {}

    def extend(self, key: RequestKey, start: int, end: int) -> int:
        if not 0 <= start <= end <= self.max_length:
            raise ValueError("Invalid context delta")
        previous = self._states.get(key)
        if start != (previous[1] if previous is not None else 0):
            raise ValueError("Context delta must extend the cached prefix")
        if previous is None:
            if not self._free:
                raise RuntimeError("Draft context cache capacity exhausted")
            slot = self._free.pop()
        else:
            slot = previous[0]
        self._states[key] = slot, end
        return slot

    def remove(self, key: RequestKey) -> None:
        previous = self._states.pop(key, None)
        if previous is not None:
            self._free.append(previous[0])

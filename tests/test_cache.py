import pytest

from disagg_sd.cache import ContextSlots
from disagg_sd.types import RequestKey


def test_context_lifecycle_and_stale_generation():
    slots = ContextSlots(2, 128)
    old, other, new = (RequestKey("r", 0), RequestKey("s", 1), RequestKey("r", 2))
    first = slots.extend(old, 0, 32)
    assert slots.extend(old, 32, 35) == first
    assert slots.extend(other, 0, 7) != first
    with pytest.raises(RuntimeError, match="capacity"):
        slots.extend(new, 0, 4)
    slots.remove(old)
    assert slots.extend(new, 0, 4) == first
    slots.remove(old)  # A late retirement cannot evict the reused slot.
    assert slots.extend(new, 4, 8) == first


def test_context_rejects_missing_or_repeated_deltas():
    slots = ContextSlots(1, 32)
    key = RequestKey("r", 0)
    with pytest.raises(ValueError, match="prefix"):
        slots.extend(key, 3, 5)
    slots.extend(key, 0, 5)
    with pytest.raises(ValueError, match="prefix"):
        slots.extend(key, 0, 5)
    with pytest.raises(ValueError, match="Invalid"):
        slots.extend(key, 5, 33)
    slots.extend(key, 5, 5)  # Another query can use the same committed anchor.

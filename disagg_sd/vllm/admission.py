"""The single scheduling compatibility seam required by this vLLM revision."""

from contextlib import contextmanager


@contextmanager
def eligible_running(scheduler, eligible: set[str]):
    """Give native scheduling eligible requests while reserving held capacity.

    Held requests keep their real KV and counters. Native preemption/admission
    still acts on the visible set. Restore order even if native scheduling fails.
    """
    original = scheduler.running
    held = [r for r in original if r.request_id not in eligible]
    scheduler.running = [r for r in original if r.request_id in eligible]
    capacity = scheduler.max_num_active_reqs
    scheduler.max_num_active_reqs = capacity - len(held)
    try:
        yield
    finally:
        survivors = {r.request_id: r for r in scheduler.running}
        survivors.update({r.request_id: r for r in held})
        scheduler.running = [r for r in original if r.request_id in survivors]
        original_ids = {r.request_id for r in original}
        scheduler.running.extend(
            r for rid, r in survivors.items() if rid not in original_ids
        )
        scheduler.max_num_active_reqs = capacity

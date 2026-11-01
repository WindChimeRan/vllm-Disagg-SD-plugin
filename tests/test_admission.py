from types import SimpleNamespace

import pytest

from disagg_sd.vllm.admission import eligible_running


@pytest.mark.parametrize("fail", [False, True])
def test_held_slots_reserved_and_native_preemption_preserved(fail):
    a, b, c, d = (SimpleNamespace(request_id=x) for x in "abcd")
    scheduler = SimpleNamespace(running=[a, b, c], max_num_active_reqs=4)
    try:
        with eligible_running(scheduler, {"a", "c"}):
            assert scheduler.running == [a, c]
            assert scheduler.max_num_active_reqs == 3
            scheduler.running.remove(a)  # native preemption
            scheduler.running.append(d)  # new arrival
            if fail:
                raise RuntimeError("native failure")
    except RuntimeError:
        assert fail
    assert scheduler.running == [b, c, d]
    assert scheduler.max_num_active_reqs == 4

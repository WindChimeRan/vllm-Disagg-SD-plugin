import pytest

from disagg_sd.policy import MineDraftPolicy, OverlapPolicy, SynchronousPolicy
from disagg_sd.runtime import Coordinator
from disagg_sd.types import DraftResult


def ready(core, rid, length=10, limit=100):
    core.add(rid, limit)
    core.admit(rid)
    core.observe(rid, length, 42, length - 1)


def complete(core, jobs):
    for job in jobs:
        core.complete_draft(
            DraftResult(job, tuple(range(job.start, job.start + job.count)))
        )


def test_minedraft_alternates_with_arrivals_and_departures():
    core = Coordinator(MineDraftPolicy(), 3, 16)
    for rid in ("a", "b", "c", "d"):
        ready(core, rid)
    warmup = core.plan(core.request_ids)
    assert {j.key.request_id for j in warmup.drafts} == {"a", "c"}
    complete(core, warmup.drafts)
    first = core.plan(core.request_ids)
    assert set(first.verify) == {"a", "c"}
    assert {j.key.request_id for j in first.drafts} == {"b", "d"}
    for rid, tokens in first.verify.items():
        core.begin_verify(rid, tokens)
        core.observe(rid, 12, 55, 11)  # correction discards the rest of this round
    core.remove("a")
    core.remove("c")
    core.end_verify(first.verify)
    complete(core, first.drafts)
    ready(core, "e")
    assert set(core.plan(core.request_ids).verify) == {"b", "d"}


def test_overlap_keeps_suffix_and_accepts_result_from_older_prefix():
    core = Coordinator(OverlapPolicy(), 3, 16)
    ready(core, "a")
    complete(core, core.plan(core.request_ids).drafts)
    plan = core.plan(core.request_ids)
    assert plan.verify == {"a": (10, 11, 12)}
    (future,) = plan.drafts
    assert future.start == 14 and future.gap == 4
    core.begin_verify("a", plan.verify["a"])
    core.observe("a", 11, 99, 10)  # first candidate rejected and corrected
    core.end_verify(["a"])
    complete(core, [future])
    retained = core.plan(core.request_ids)
    assert retained.verify == {"a": (11, 12)}
    assert not retained.drafts
    core.begin_verify("a", retained.verify["a"])
    core.observe("a", 14, 88, 13)  # suffix accepted, target fills bonus at 13
    core.end_verify(["a"])
    assert core.plan(core.request_ids).verify == {"a": (14, 15, 16)}


@pytest.mark.parametrize("reset", [True, False])
def test_late_result_cannot_attach_to_reused_id(reset):
    core = Coordinator(OverlapPolicy(), 3, 16)
    ready(core, "a")
    (old,) = core.plan(core.request_ids).drafts
    if reset:
        core.reset("a")
    else:
        core.remove("a")
        core.add("a", 100)
    core.admit("a")
    core.observe("a", 10, 42, 9)
    assert not core.complete_draft(DraftResult(old, (1, 2, 3)))
    (new,) = core.plan(core.request_ids).drafts
    assert new.key != old.key


def test_limits_and_contiguous_verification():
    core = Coordinator(OverlapPolicy(), 3, 16)
    ready(core, "a", limit=12)
    (job,) = core.plan(core.request_ids).drafts
    assert job.count == 2
    with pytest.raises(ValueError, match="Malformed"):
        core.complete_draft(DraftResult(job, (1,)))
    complete(core, [job])
    with pytest.raises(ValueError, match="contiguously"):
        core.begin_verify("a", (11,))
    core.begin_verify("a", (10,))  # native budget can truncate a draft window
    core.observe("a", 12, 88, 11)
    core.end_verify(["a"])
    assert not core.plan(core.request_ids).drafts


def test_minedraft_single_remaining_cohort_progresses():
    core = Coordinator(MineDraftPolicy(), 3, 16)
    ready(core, "a")
    ready(core, "b")
    core.remove("a")
    (job,) = core.plan(core.request_ids).drafts
    assert job.key.request_id == "b"
    complete(core, [job])
    core.begin_verify("b", (10, 11, 12))
    core.observe("b", 11, 1, 10)
    core.end_verify(["b"])
    next_plan = core.plan(core.request_ids)
    assert not next_plan.verify
    assert next_plan.drafts[0].start == 11


def test_synchronous_control_does_not_overlap_decode_and_drafting():
    core = Coordinator(SynchronousPolicy(), 3, 16)
    ready(core, "a")
    ready(core, "b")
    plan = core.plan(core.request_ids)
    assert not plan.verify and len(plan.drafts) == 2
    complete(core, plan.drafts)
    ready(core, "c")
    plan = core.plan(core.request_ids)
    assert set(plan.verify) == {"a", "b"}
    assert not plan.drafts


def test_minedraft_balances_admitted_requests_after_queued_cancellation():
    core = Coordinator(MineDraftPolicy(), 3, 16)
    for rid in "abcd":
        core.add(rid, 100)
    assert not core.plan(core.request_ids).drafts
    for rid in "ab":
        assert core.admit(rid)
        core.observe(rid, 10, 42, 9)
    core.remove("a")  # cohort A finishes while c and d are still queued
    core.remove("c")  # queued cancellation must not change active balance
    assert core.admit("d")
    assert not core.admit("d")  # further prefill chunks are not new admissions
    core.observe("d", 10, 42, 9)
    warmup = core.plan(core.request_ids)
    assert [job.key.request_id for job in warmup.drafts] == ["d"]
    complete(core, warmup.drafts)
    plan = core.plan(core.request_ids)
    assert set(plan.verify) == {"d"}
    assert [job.key.request_id for job in plan.drafts] == ["b"]


def test_preemption_waits_for_readmission_and_releases_cohort():
    core = Coordinator(MineDraftPolicy(), 3, 16)
    ready(core, "a")
    ready(core, "b")
    (old,) = core.plan(core.request_ids).drafts
    core.reset("a")
    assert not core.complete_draft(DraftResult(old, (1, 2, 3)))
    with pytest.raises(ValueError, match="Admit"):
        core.observe("a", 10, 42, 9)
    ready(core, "c")  # fills the active cohort vacated by a
    assert [j.key.request_id for j in core.plan(core.request_ids).drafts] == ["c"]
    assert core.admit("a")
    core.observe("a", 10, 42, 9)


def test_empty_running_set_resets_phase_despite_waiting_requests():
    core = Coordinator(MineDraftPolicy(), 3, 16)
    ready(core, "a")
    core.add("waiting", 100)
    complete(core, core.plan(["a"]).drafts)
    tokens = core.plan(["a"]).verify["a"]
    core.begin_verify("a", tokens)
    core.remove("a")
    core.end_verify(["a"])
    core.admit("waiting")
    core.observe("waiting", 10, 42, 9)
    ready(core, "new")
    assert [j.key.request_id for j in core.plan(core.request_ids).drafts] == ["waiting"]

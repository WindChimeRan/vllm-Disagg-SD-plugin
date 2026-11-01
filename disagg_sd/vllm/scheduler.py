from vllm.v1.core.sched.scheduler import Scheduler

from disagg_sd.vllm.admission import eligible_running
from disagg_sd.vllm.session import get_session


class DisaggScheduler(Scheduler):
    def __init__(self, vllm_config, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)
        self.session = get_session(vllm_config)
        self.session.serving = True

    def add_request(self, request):
        params = request.sampling_params
        if (
            params is None
            or params.temperature != 0
            or request.lora_request
            or request.structured_output_request is not None
            or request.resumable
            or request.mm_features
        ):
            raise ValueError("Use greedy, text-only, non-streaming requests")
        super().add_request(request)
        limit = min(request.num_prompt_tokens + request.max_tokens, self.max_model_len)
        self.session.coordinator.add(request.request_id, limit)
        self.session.log("add", request=request.request_id)

    def schedule(self, throttle_prefills=False):
        session = self.session
        core = session.coordinator
        session.receive()
        while True:
            active = [r.request_id for r in self.running]
            plan = core.plan(active, can_draft=not session.worker.busy)
            session.submit(plan.drafts)
            prefill = {
                r.request_id for r in self.running if not core.ready(r.request_id)
            }
            can_admit = (
                bool(self.waiting) and len(self.running) < self.max_num_active_reqs
            )
            if plan.verify or prefill or can_admit or not session.worker.busy:
                break
            session.receive(wait=True)

        for request in self.running:
            request.spec_token_ids = list(plan.verify.get(request.request_id, ()))
        with eligible_running(self, set(plan.verify) | prefill):
            output = super().schedule(throttle_prefills)
        for rid in output.num_scheduled_tokens:
            if core.admit(rid):
                session.log("admit", request=rid, generation=core.key(rid).generation)
            if rid in plan.verify:
                core.begin_verify(
                    rid, tuple(output.scheduled_spec_decode_tokens.get(rid, ()))
                )
        session.log(
            "target_schedule",
            requests=list(output.num_scheduled_tokens),
            candidates={
                rid: len(ts) for rid, ts in output.scheduled_spec_decode_tokens.items()
            },
        )
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        session = self.session
        capture = session.capture
        session.capture = None
        output = super().update_from_output(scheduler_output, model_runner_output)
        for rid in scheduler_output.num_scheduled_tokens:
            index = model_runner_output.req_id_to_index[rid]
            proposed = len(scheduler_output.scheduled_spec_decode_tokens.get(rid, ()))
            sampled = len(model_runner_output.sampled_token_ids[index])
            session.log(
                "target_result",
                request=rid,
                proposed=proposed,
                accepted=min(proposed, max(0, sampled - 1)),
            )
        if capture is not None:
            for span in capture.spans:
                rid = span.request_id
                request = self.requests.get(rid)
                if request is None or request.is_finished():
                    continue
                key = session.coordinator.key(rid)
                start = span.start
                end = request.num_computed_tokens
                offset = span.offset
                if not start <= end <= start + span.count:
                    raise RuntimeError("Native committed feature span is inconsistent")
                session.features.append(
                    key,
                    start,
                    [x[offset : offset + end - start] for x in capture.layers],
                )
                session.coordinator.observe(
                    rid,
                    request.num_tokens,
                    request.all_token_ids[-1] if request.num_output_tokens else None,
                    session.features.length(key),
                )
                index = model_runner_output.req_id_to_index[rid]
                session.log(
                    "target_commit",
                    request=rid,
                    length=request.num_tokens,
                    emitted=len(model_runner_output.sampled_token_ids[index]),
                )
        elif scheduler_output.total_num_scheduled_tokens:
            raise RuntimeError("Target worker did not supply auxiliary hidden states")
        session.coordinator.end_verify(scheduler_output.num_scheduled_tokens)
        return output

    def _preempt_request(self, request, timestamp, drop_stale_output=False):
        session = self.session
        session.retire(session.coordinator.key(request.request_id))
        session.coordinator.reset(request.request_id)
        session.log("preempt", request=request.request_id)
        return super()._preempt_request(request, timestamp, drop_stale_output)

    def _free_request(self, request, delay_free_blocks=False):
        self.session.remove(request.request_id)
        return super()._free_request(request, delay_free_blocks)

    def shutdown(self):
        self.session.close()
        super().shutdown()

"""Exercise arrivals, mixed lengths, and aborts through vLLM's public engine API."""

import argparse
import json
from pathlib import Path

from vllm import EngineArgs, LLMEngine, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft")
    parser.add_argument(
        "--policy",
        choices=["minedraft", "overlap", "sync", "target"],
        default="minedraft",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Save the target-process trace; use Nsight Systems for both processes",
    )
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--long-prefill", action="store_true")
    parser.add_argument(
        "--reset-at",
        type=int,
        help="Force native preemption and replay at this engine step",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    extra = {}
    if args.policy != "target":
        if not args.draft:
            parser.error("--draft is required for speculative decoding")
        extra = dict(
            worker_cls="disagg_sd.vllm.worker.DisaggWorker",
            scheduler_cls="disagg_sd.vllm.scheduler.DisaggScheduler",
            speculative_config={
                "method": "disagg_dflash",
                "model": args.draft,
                "num_speculative_tokens": 15,
                "speculator_options": {
                    "policy": args.policy,
                    "trace_path": str(args.output / "events.jsonl"),
                },
            },
        )
    engine = LLMEngine.from_engine_args(
        EngineArgs(
            model=args.target,
            dtype="bfloat16",
            max_model_len=1024,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=512,
            gpu_memory_utilization=0.15,
            enforce_eager=True,
            enable_prefix_caching=False,
            async_scheduling=False,
            distributed_executor_backend="uni",
            disable_log_stats=True,
            **extra,
        )
    )
    prompts = [
        'def fibonacci(n):\n    """Return the nth Fibonacci number."""\n',
        "Explain why the sky is blue in three sentences.\n",
        'def is_palindrome(text):\n    """Ignore spaces and case."""\n',
        "The first ten prime numbers are",
        "Write a Python function to merge two sorted lists.\n",
        "A short story about a robot learning to garden:\n",
        "def binary_search(values, target):\n",
        "Explain what a database transaction does.\n",
    ]
    if args.long_prefill:
        prompts[7] = "This is background context. " * 110 + prompts[7]
    # Admission is tied to engine steps, not acceptance or model outputs.
    arrivals = {0: [0, 1, 2, 3], 2: [4, 5], 5: [6, 7]}
    final = {}
    outputs = {}
    profiler = None
    if args.profile:
        import torch

        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )
        profiler.start()
    try:
        step = 0
        while step <= 5 or engine.has_unfinished_requests():
            if step > 1000:
                raise RuntimeError("Engine made insufficient progress")
            for i in arrivals.get(step, []):
                engine.add_request(
                    str(i),
                    prompts[i],
                    SamplingParams(
                        temperature=0, max_tokens=32 + 8 * i, ignore_eos=True
                    ),
                )
            if step == 7:
                engine.abort_request(["5"])
            if step == args.reset_at:
                engine.reset_prefix_cache(reset_running_requests=True)
            for result in engine.step():
                outputs[result.request_id] = list(result.outputs[0].token_ids)
                if result.finished:
                    final[result.request_id] = outputs[result.request_id]
            step += 1
        if set(final) != {str(i) for i in range(8) if i != 5}:
            raise AssertionError(f"Unexpected completed requests: {set(final)}")
        (args.output / "tokens.json").write_text(json.dumps(final, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "policy": args.policy,
                    "steps": step,
                    "completed": len(final),
                    "tokens": sum(map(len, final.values())),
                }
            )
        )
    finally:
        if profiler is not None:
            profiler.stop()
            profiler.export_chrome_trace(str(args.output / "target-profile.json"))
        engine.engine_core.shutdown()


if __name__ == "__main__":
    main()

"""Check token parity and report measured CUDA kernel overlap, if profiled."""

import argparse
import json
from pathlib import Path


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def overlap(left, right):
    i = j = 0
    total = 0
    while i < len(left) and j < len(right):
        a, b = left[i], right[j]
        total += max(0, min(a[1], b[1]) - max(a[0], b[0]))
        if a[1] <= b[1]:
            i += 1
        else:
            j += 1
    return total


def check(reference: Path, candidate: Path):
    expected = json.loads((reference / "tokens.json").read_text())
    actual = json.loads((candidate / "tokens.json").read_text())
    if actual != expected:
        differing = [rid for rid in expected if actual.get(rid) != expected[rid]]
        raise AssertionError(f"Token mismatch: {differing}")
    result = {
        "run": str(candidate),
        "matching_tokens": sum(map(len, expected.values())),
    }
    trace_path = candidate / "profile.json"
    if trace_path.exists():
        events = json.loads(trace_path.read_text())["traceEvents"]
        devices = {}
        model_devices = {}
        for event in events:
            if event.get("cat") == "kernel":
                device = event["args"]["device"]
                interval = (event["ts"], event["ts"] + event["dur"])
                devices.setdefault(device, []).append(interval)
                # Recognize GEMM and FlashAttention names used by the pinned
                # build, including Blackwell's cuBLAS nvjet kernels.
                if any(
                    name in event["name"].lower()
                    for name in (
                        "gemm",
                        "matmul",
                        "nvjet",
                        "fmha",
                        "flash_fwd",
                        "attention",
                    )
                ):
                    model_devices.setdefault(device, []).append(interval)
        if len(devices) != 2:
            raise AssertionError(f"Expected two profiled GPUs, saw {list(devices)}")
        left, right = (merge(x) for _, x in sorted(devices.items()))
        result["kernel_overlap_ms"] = overlap(left, right) / 1000
        if result["kernel_overlap_ms"] <= 0:
            raise AssertionError("No simultaneous target/draft CUDA kernels observed")
        if len(model_devices) != 2:
            raise AssertionError("Cannot identify model kernels on both GPUs")
        left, right = (merge(x) for _, x in sorted(model_devices.items()))
        result["model_kernel_overlap_ms"] = overlap(left, right) / 1000
        if result["model_kernel_overlap_ms"] <= 0:
            raise AssertionError("No simultaneous GEMM/attention kernels observed")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", nargs="+", type=Path)
    args = parser.parse_args()
    for candidate in args.candidates:
        check(args.reference, candidate)

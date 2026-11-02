#!/usr/bin/env python3
"""Fixed-input RGDR public-call diagnostics; does not replace ATK acceptance.

No torch/NPU imports until main, so --help and control tests work offline.
Event intervals include stream idle time and must not be called kernel duration.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time


OP_REL = Path("tests/atk/recurrent_gated_delta_rule")
CASE_NAME = "atk_recurrent_gated_delta_rule_perf.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_case(path, case_id):
    cases = [c for c in json.loads(Path(path).read_text(encoding="utf-8")) if c["id"] == case_id]
    if len(cases) != 1:
        raise ValueError("perf case id must identify exactly one original case")
    specs = [x["range_values"] for x in cases[0]["inputs"] if x["name"] == "case_spec"]
    if len(specs) != 1:
        raise ValueError("expected one original case_spec")
    spec = json.loads(specs[0])
    expected = {"B": 1, "T": 1, "HK": 1, "HV": 1, "K": 128, "V": 128,
                "state_dtype": "bf16", "gate_mode": "g"}
    if any(spec.get(k) != v for k, v in expected.items()):
        raise ValueError("this diagnostic only supports the confirmed default A5 perf shape")
    if (spec.get("repeat_calls", 1) != 1 or spec.get("state_layout", "contiguous") != "contiguous"
            or spec.get("input_layout", "contiguous") != "contiguous" or spec.get("accepted_tokens") is not None):
        raise ValueError("requires one call, contiguous inputs/state and no accepted-token input")
    return spec


def summarize(values):
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("timing samples must be nonempty, finite and nonnegative")
    ordered = sorted(values)

    def percentile(q):
        index = (len(ordered) - 1) * q
        lower = math.floor(index)
        upper = math.ceil(index)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {"count": len(values), "mean_us": statistics.mean(values),
            "median_us": statistics.median(values), "std_us": statistics.pstdev(values),
            "min_us": min(values), "max_us": max(values), "p90_us": percentile(.9), "p99_us": percentile(.99)}


def measure_wall(call, prepare, synchronize, clock=time.perf_counter_ns):
    prepare()  # Includes state restore + completion; neither is timed.
    start = clock()
    output = call()
    returned = clock()
    synchronize()
    completed = clock()
    # Hold the returned tensor through completion; never D2H/check it here.
    return output, {"host_submit_us": (returned - start) / 1000,
                    "call_and_sync_us": (completed - start) / 1000,
                    "post_return_wait_us": (completed - returned) / 1000}


def measure_event(call, prepare, synchronize, start, end, stream):
    prepare()
    start.record(stream)
    output = call()
    end.record(stream)
    synchronize()
    return output, {"stream_event_us": start.elapsed_time(end) * 1000}


@contextmanager
def npu_deadline():
    # Default SIGALRM terminates even when a native sync does not return.
    # Covers preparation/submission/sync; does not identify kernel entry time.
    previous = signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(60)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def target_objects(root):
    result = {}
    root = Path(root)
    for path in sorted(root.rglob("*.o")):
        name = path.relative_to(root).as_posix()
        if "recurrentgateddeltarule" in name.lower().replace("_", ""):
            result[name] = sha256(path)
    if not result:
        raise RuntimeError(f"no RGDR kernel .o evidence under {root}")
    return result


def compare_objects(runtime_root, built_root):
    runtime, built = target_objects(runtime_root), target_objects(built_root)
    if runtime != built:
        raise RuntimeError("installed/build RGDR .o paths or hashes differ; rebuild/install the selected branch")
    return built


def validate_opapi_check(text, expected):
    prefix = "loaded libcust_opapi.so: "
    paths = [Path(line[len(prefix):]).resolve() for line in text.splitlines() if line.startswith(prefix)]
    if (paths != [Path(expected).resolve()] or
            "[OK] loaded libcust_opapi.so matches the freshly built one." not in text):
        raise RuntimeError("op_api check did not prove this installed package's library; see artifact_check.log")


def summarize_profile(directory, expected_calls):
    # Keep each exported file separate: duplicate views are not extra samples.
    evidence = []
    for path in sorted(Path(directory).rglob("*op_summary*.csv")):
        groups = defaultdict(list)
        with path.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                kind = row.get("OP Type") or row.get("Op Type") or "unknown"
                if row.get("Task Duration(us)"):
                    groups[kind].append(float(row["Task Duration(us)"]))
        targets = [k for k in groups if k.lower().replace("_", "") == "recurrentgateddeltarule"]
        if not targets:
            continue
        if len(targets) != 1 or len(groups[targets[0]]) != expected_calls:
            raise RuntimeError(f"unexpected RGDR task count in {path}; inspect the retained profile")
        evidence.append({"file": str(path), "tasks": {k: summarize(v) for k, v in groups.items()}})
    if not evidence:
        raise RuntimeError("no RGDR op_summary Task Duration evidence; unprofiled timings are retained")
    return evidence


def profile_calls(torch, torch_npu, call_with_state, initial_state, synchronize, count, output):
    # All allocations/resets finish BEFORE starting the profiler.
    with npu_deadline():
        states = [initial_state.clone() for _ in range(count)]
        synchronize()
    profiler = torch_npu.profiler
    results = []
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        experimental_config=profiler._ExperimentalConfig(
            profiler_level=profiler.ProfilerLevel.Level0, data_simplification=False),
        record_shapes=True, profile_memory=False, with_stack=False,
        on_trace_ready=profiler.tensorboard_trace_handler(str(output)),
    ) as capture:
        for state in states:
            with npu_deadline(), torch.profiler.record_function("RGDR_PUBLIC_CALL"):
                results.append(call_with_state(state))
                synchronize()
            capture.step()
    return results, states


def write_result(path, result):
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def git_value(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True, encoding="utf-8").strip()


def run(args, root, output, report):
    # Configure the installed package OPP before torch_npu initializes the device.
    import fla_npu

    vendor = Path(fla_npu._prepare_embedded_opp())
    opapi = Path(fla_npu._resolve_packaged_opapi(vendor))
    check = subprocess.run(
        [sys.executable, str(root / "scripts/verify_libcust_opapi_md5.py"),
         "--run-package", str(args.run_package.resolve()), "--no-kernel"],
        cwd=root, capture_output=True, text=True, timeout=360,
    )
    (output / "artifact_check.log").write_text(check.stdout + check.stderr, encoding="utf-8")
    if check.returncode:
        raise RuntimeError("op_api artifact check failed; see artifact_check.log")
    validate_opapi_check(check.stdout, opapi)
    built = args.built_kernel or root / "build/lib/fla_npu/opp/vendors/fla_npu_transformer/op_impl/ai_core/tbe/kernel"
    objects = compare_objects(vendor / "op_impl/ai_core/tbe/kernel", built)

    import torch
    import torch_npu
    from fla_npu.ops import ascendc
    from fla_npu.ops.ascendc import _stable

    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    if "950" not in torch.npu.get_device_name(args.device):
        raise RuntimeError("this diagnostic requires the confirmed A5 / Ascend950 hardware")
    sys.path.insert(0, str(root / OP_REL))
    from executor_recurrent_gated_delta_rule import build_inputs

    case_path = root / OP_REL / CASE_NAME
    spec = load_case(case_path, args.case_id)
    stream = torch.npu.current_stream()
    synchronize = stream.synchronize
    with torch.inference_mode():
        with npu_deadline():
            inputs = build_inputs(spec, device)
            initial_state = inputs["state"].clone()
            synchronize()
        kwargs = {k: inputs[k] for k in ("beta", "scale", "actual_seq_lengths", "ssm_state_indices",
                                         "num_accepted_tokens", "g", "gk")}

        def call_with_state(state):
            return ascendc.recurrent_gated_delta_rule(
                inputs["query"], inputs["key"], inputs["value"], state, **kwargs)

        def call():
            return call_with_state(inputs["state"])

        def prepare():
            inputs["state"].copy_(initial_state)
            synchronize()

        def finite_check(result, state):
            # Outside all timing/profiler regions; sanity only, not CPU accuracy.
            if not torch.isfinite(result.float()).all().item() or not torch.isfinite(state.float()).all().item():
                raise RuntimeError("nonfinite output/state in an untimed sanity check")

        report.update(status="warming_up", label=args.label, case_spec=spec,
                      git_commit=git_value(root, "rev-parse", "HEAD"), device=args.device,
                      package=str(Path(fla_npu.__file__).resolve()), rgdr_kernel_sha256=objects)
        write_result(output / "result.json", report)
        for _ in range(args.warmup):
            with npu_deadline():
                prepare()
                result = call()
                synchronize()
        finite_check(result, inputs["state"])
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        # Eagerly initialize both lazily-created event handles outside timing.
        with npu_deadline():
            start.record(stream)
            end.record(stream)
            synchronize()
        report["status"] = "measuring"
        write_result(output / "result.json", report)
        samples = defaultdict(list)
        for mode in ("wall", "event", "empty_wall", "empty_event"):
            selected = call if not mode.startswith("empty") else lambda: None
            for _ in range(args.samples):
                with npu_deadline():
                    if mode.endswith("wall"):
                        result, values = measure_wall(selected, prepare, synchronize)
                    else:
                        result, values = measure_event(selected, prepare, synchronize, start, end, stream)
                for key, value in values.items():
                    samples[("empty_" if mode.startswith("empty") else "") + key].append(value)
            if not mode.startswith("empty"):
                finite_check(result, inputs["state"])

        source_files = [root / OP_REL / name for name in (CASE_NAME, "executor_recurrent_gated_delta_rule.py")]
        report.update({
            "status": "timings_complete", "label": args.label, "case_spec": spec,
            "git_commit": git_value(root, "rev-parse", "HEAD"),
            "git_branch": git_value(root, "branch", "--show-current"),
            "git_status": git_value(root, "status", "--porcelain"),
            "python": sys.executable, "torch": torch.__version__, "torch_npu": torch_npu.__version__,
            "environment": {k: os.environ.get(k) for k in (
                "ASCEND_HOME_PATH", "ASCEND_LAUNCH_BLOCKING", "TASK_QUEUE_ENABLE",
                "FLA_NPU_STABLE_ABI", "FLA_NPU_STABLE_STREAM", "FLA_NPU_STABLE_VALIDATE")},
            "device": args.device, "device_name": torch.npu.get_device_name(args.device),
            "stream": int(stream.npu_stream), "package": str(Path(fla_npu.__file__).resolve()),
            "opapi": str(opapi), "opapi_sha256": sha256(opapi),
            "public_function_module": ascendc.recurrent_gated_delta_rule.__module__,
            "stable": _stable._loaded_path,
            "stable_sha256": sha256(_stable._loaded_path) if _stable._loaded_path else None,
            "run_package": str(args.run_package.resolve()), "run_package_sha256": sha256(args.run_package),
            "built_kernel": str(Path(built).resolve()), "rgdr_kernel_sha256": objects,
            "source_sha256": {str(p.relative_to(root)): sha256(p) for p in source_files},
            "warmup": args.warmup, "samples_us": dict(samples),
            "torch_mode": "inference_mode", "profile_requested": args.profile,
            "profile_samples": args.profile_samples if args.profile else 0,
            "statistics": {k: summarize(v) for k, v in samples.items()},
            "scope": "fixed inputs; reset/checks outside timing; event includes stream idle; no overhead subtraction",
            "accuracy": "untimed finite sanity only; no CPU accuracy run",
        })
        write_result(output / "result.json", report)
        for key, stats in report["statistics"].items():
            print(f"{key}: n={stats['count']} mean={stats['mean_us']:.5f} median={stats['median_us']:.5f} "
                  f"std={stats['std_us']:.5f} p90={stats['p90_us']:.5f}", flush=True)
        if args.profile:
            results, states = profile_calls(torch, torch_npu, call_with_state, initial_state, synchronize,
                                           args.profile_samples, output / "profile")
            finite_check(results[-1], states[-1])
            report["system_profile"] = summarize_profile(output / "profile", args.profile_samples)
            for evidence in report["system_profile"]:
                print("profile:", json.dumps(evidence, ensure_ascii=False), flush=True)
        report["status"] = "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, help="required when running a copy outside the checkout")
    parser.add_argument("--run-package", type=Path, required=True, help="this build's exact .run, not a wheel")
    parser.add_argument("--built-kernel", type=Path, help="override staged kernel root, not the binary/bin directory")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case-id", type=int, default=0)
    parser.add_argument("--label", default="scratch")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--profile", action="store_true", help="separate system profile AFTER unprofiled measurements")
    parser.add_argument("--profile-samples", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.repo_root or next((p for p in Path(__file__).resolve().parents if (p / OP_REL / CASE_NAME).is_file()), None)
    if root is None or not (root / OP_REL / CASE_NAME).is_file():
        parser.error("--repo-root must point to the selected branch checkout")
    if args.run_package.suffix != ".run" or not args.run_package.is_file():
        parser.error("--run-package must be an existing .run from this build")
    if min(args.warmup, args.samples, args.profile_samples) <= 0 or args.device < 0:
        parser.error("warmup/samples/profile-samples must be positive; device must be nonnegative")
    if not hasattr(signal, "SIGALRM"):
        parser.error("online collection requires Linux SIGALRM watchdog support")
    root = root.resolve()
    output = (args.output or root / OP_REL / "runs/public_latency" /
              datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "output": str(output)}
    write_result(output / "result.json", report)
    print(f"output={output}", flush=True)
    try:
        run(args, root, output, report)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_result(output / "result.json", report)


if __name__ == "__main__":
    main()

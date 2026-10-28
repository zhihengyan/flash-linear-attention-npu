#!/usr/bin/env python3
"""Collect the existing RGDR perf case through its normal and private timer paths."""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone

import parse_timer_csv as timer_parser
import trace_parser

ROOT = Path(__file__).resolve().parents[4]
OP_DIR = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_spec(path, case_id):
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    case = next(c for c in cases if c["id"] == case_id)
    return json.loads(next(x["range_values"] for x in case["inputs"] if x["name"] == "case_spec"))


def prepare_opp():
    # Resolve the installed top-level package without importing torch/torch_npu/fla_npu.
    spec = importlib.util.find_spec("fla_npu")
    if spec is None or not spec.origin:
        raise RuntimeError("install this branch's complete wheel before collection")
    package = Path(spec.origin).resolve().parent
    vendor = package / "opp/vendors/fla_npu_transformer"
    lib = vendor / "op_api/lib/libcust_opapi.so"
    if not lib.is_file():
        raise RuntimeError(f"missing embedded OPP: {lib}")
    paths = [str(vendor), str(vendor.parent.parent)]
    paths += [x for x in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep) if x]
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(dict.fromkeys(paths))
    os.environ["FLA_NPU_OP_API_LIB"] = str(lib)
    return package, lib


def check_python_environment():
    atk_env = os.environ.get("ATK_ENV", "")
    if atk_env and Path(atk_env).resolve() != Path(sys.prefix).resolve():
        raise RuntimeError("activate ATK_ENV before this script so ATK and capture use the same Python environment")
    if importlib.util.find_spec("atk") is None:
        raise RuntimeError("the active Python must contain the existing ATK package; activate the original test environment")


def run_logged(command, log, env=None, timeout=None):
    with log.open("w", encoding="utf-8") as output:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, timeout=timeout)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log}")
    return log.read_text(encoding="utf-8", errors="replace")


def verify_atk_stage(directory, stage):
    # ATK may log a caught exception and still exit 0. Check the actual task report.
    evidence = []
    for log in sorted(directory.rglob("atk.log")):
        content = log.read_text(encoding="utf-8", errors="replace")
        if "atk task failed!" in content or "All cases is running failed." in content:
            raise RuntimeError(f"ATK {stage} task failed; see {log}")
        reports = re.findall(r"Total Task:\s*(\d+),\s*success\s+(\d+),\s*failed\s+(\d+)", content)
        if not reports:
            continue
        total, success, failed = map(int, reports[-1])
        if (total, success, failed) != (1, 1, 0):
            raise RuntimeError(f"ATK {stage} must execute exactly one successful case; "
                               f"got total={total}, success={success}, failed={failed}; see {log}")
        if stage == "accuracy":
            results = re.findall(r"\bacc_pass_result:\s*(\S+)", content)
            if not results or any(result != "Pass" for result in results):
                raise RuntimeError(f"ATK accuracy requires acc_pass_result:Pass for every reported result; see {log}")
        evidence.append(str(log))
    if not evidence:
        raise RuntimeError(f"ATK {stage} has no task-count evidence; inspect {directory}")
    return evidence


def verify_target_kernel_objects(runtime_root, built_root):
    def objects(root):
        return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob("*.o")
                if "recurrentgateddeltarule" in str(path.relative_to(root)).lower().replace("_", "")}
    runtime = objects(runtime_root)
    built = objects(built_root)
    if not runtime or not built or runtime != built:
        raise ValueError("this operator's runtime/built kernel .o sets are empty or differ")
    return {"runtime_root": str(runtime_root), "built_root": str(built_root), "sha256": runtime}


def summarize_system_profile(directory):
    summaries = []
    found_target = False
    for path in sorted(directory.rglob("*op_summary*.csv")):
        totals = {}
        with path.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                if "Task Duration(us)" not in row:
                    continue
                name = row.get("Op Name") or row.get("Name") or row.get("OP Type") or row.get("Op Type") or "unknown"
                signature = "".join(str(v) for v in row.values()).lower().replace("_", "")
                found_target |= "recurrentgateddeltarule" in signature
                try:
                    value = float(row["Task Duration(us)"])
                except (TypeError, ValueError):
                    continue
                totals.setdefault(name, []).append(value)
        if totals:
            summaries.append({"file": str(path), "tasks": {
                name: {"count": len(values), "mean_us": sum(values) / len(values),
                       "min_us": min(values), "max_us": max(values)}
                for name, values in totals.items()}})
    if not found_target:
        raise RuntimeError("no RGDR op_summary Task Duration evidence; retain ATK logs and inspect profile output")
    return summaries


def parse_capture(raw_path, output, cycles_per_us=None, shared_clock=False):
    raw = timer_parser.load_raw_data(raw_path)
    metadata = timer_parser.validate_metadata(raw)
    parsed = output / "rank_0_npu_time_parsed.csv"
    timer_parser.parse_and_dump(raw_path, parsed, cycles_per_us)
    timer_parser.print_summary(parsed)
    if cycles_per_us is not None and shared_clock:
        events = trace_parser.generate_trace_json(parsed, 0, trace_parser.find_min_start(parsed),
                                                  cycles_per_us, shared_clock=True)
        write_json(output / "rank_0_npu_time.json", events)
        metadata["trace"] = "rank_0_npu_time.json"
    else:
        metadata["trace"] = "not generated: verified frequency and shared-clock assertion required"
    metadata["units"] = "cycles" if cycles_per_us is None else "cycles and us"
    write_json(output / "timer_metadata.json", metadata)
    return metadata


def validate_capture_case(case, metadata):
    """Check actual per-core calls against the original weighted head scheduler."""
    for name in ("B", "T", "HK", "HV"):
        if metadata[name] != case[name]:
            raise ValueError(f"timer {name} does not match the original perf case")
    state_bytes = 4 if case["state_dtype"] == "fp32" else 2
    step = metadata["v_step"]
    cores = metadata["block_dim"]
    if (metadata["state_bytes"] != state_bytes or not 8 <= step <= 128 or step % 8 or
            (metadata["state_out_buffers"], metadata["attn_out_buffers"]) not in ((1, 1), (1, 2), (2, 2)) or
            cores > case["B"] * case["HV"]):
        raise ValueError("invalid runtime state dtype/tiling/core metadata")
    expected = [dict.fromkeys(timer_parser.DYNAMIC_TIMING_NAMES.values(), 0) for _ in range(cores)]
    lengths = case["seq_lengths"]
    average = (sum(lengths) * case["HV"] + cores - 1) // cores
    load = used = 0
    slices = (128 + step - 1) // step
    has_g = case["gate_mode"] in ("g", "both")
    has_gk = case["gate_mode"] in ("gk", "both")
    for length in lengths:
        copied = set()
        for _ in range(case["HV"] if length > 0 else 0):
            core = used
            if core >= cores:
                raise ValueError("scheduler assigned a work item outside the launched core set")
            counts = expected[core]
            if core not in copied:
                for name in ("BATCH_GATES_ISSUE", "BETA_DEQUEUE_WAIT", "BETA_CAST_ISSUE"):
                    counts[name] += 1
                if has_g:
                    counts["G_DEQUEUE_WAIT"] += 1
                    counts["G_EXP_ISSUE"] += 1
                copied.add(core)
            for name in ("HEAD_TOTAL_ISSUE", "HEAD_QKV_ISSUE", "Q_COPY_WAIT", "K_COPY_WAIT", "V_COPY_WAIT",
                         "QKV_CAST_ISSUE", "QUERY_SCALE_ISSUE"):
                counts[name] += 1
            if has_gk:
                counts["GK_COPY_WAIT"] += 1
                counts["GK_EXP_ISSUE"] += 1
            for name in ("STATE_PREFETCH_ISSUE", "STATE_DEQUEUE_WAIT", "STATE_CAST_COPY_ISSUE", "SLICE_TOTAL_ISSUE"):
                counts[name] += slices
            for name in ("TOKEN_PARAMS", "OUTPUT_ALLOC_WAIT", "RECURRENT_VF_ISSUE", "OUTPUT_ENQUEUE",
                         "ATTN_DEQUEUE_WAIT", "ATTN_COPY_ISSUE", "STATE_OUT_DEQUEUE_WAIT", "STATE_OUT_COPY_ISSUE",
                         "TOKEN_TOTAL_ISSUE"):
                counts[name] += slices * length
            load += length
            if load >= average:
                load = 0
                used += 1
    if metadata["actual_calls_per_core"] != expected:
        raise ValueError("actual timer phase counts differ from the selected case/scheduler; missing work or instrumentation")
    return expected


def guarded_npu_call(function, synchronize):
    # A conservative diagnostic call+synchronize watchdog, including host submission.
    # It is not proof of the instant the kernel entered the device.
    old_handler = signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(60)
    try:
        value = function()
        synchronize()
        return value
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def capture_worker(args, output):
    package, opapi = prepare_opp()
    import torch
    import torch_npu  # noqa: F401 -- OPP was configured before backend initialization.
    from fla_npu.ops import ascendc
    from fla_npu.ops.ascendc import _stable

    torch.npu.set_device(args.device)
    _stable.load()
    # Validate the installed launcher against the current checkout, not only its own wheel stamp.
    path = ROOT / "torch_custom/fla_npu/csrc/build_stable.py"
    spec = importlib.util.spec_from_file_location("rgdr_current_build", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    loaded = ctypes.CDLL(_stable._loaded_path)
    loaded.fla_npu_stable_source_hash.restype = ctypes.c_char_p
    actual_hash = loaded.fla_npu_stable_source_hash().decode()
    if actual_hash != builder.source_hash():
        raise RuntimeError("installed stable adapter does not match this checkout")
    sys.path.insert(0, str(OP_DIR))
    from executor_recurrent_gated_delta_rule import build_inputs

    case = load_spec(OP_DIR / "atk_recurrent_gated_delta_rule_perf.json", args.case_id)
    if int(case.get("repeat_calls", 1)) != 1:
        raise RuntimeError("this collection entry captures one launch; choose the original single-launch perf case")
    inputs = build_inputs(case, torch.device(f"npu:{args.device}"))
    saved = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    torch.save(saved, output / "inputs.pt")
    write_json(output / "input_layouts.json", {
        name: {"shape": list(value.shape), "stride": list(value.stride()),
               "storage_offset": value.storage_offset(), "dtype": str(value.dtype)}
        for name, value in inputs.items() if isinstance(value, torch.Tensor)})
    op = _stable._op("_profile_recurrent_gated_delta_rule")
    kwargs = {name: inputs[name] for name in (
        "beta", "scale", "actual_seq_lengths", "ssm_state_indices", "num_accepted_tokens", "g", "gk")}

    def normal(state):
        return ascendc.recurrent_gated_delta_rule(inputs["query"], inputs["key"], inputs["value"], state, **kwargs)

    def state_copy():
        source = inputs["state"]
        storage = torch.empty(source.untyped_storage().nbytes() // source.element_size(),
                              dtype=source.dtype, device=source.device)
        copied = torch.as_strided(storage, source.shape, source.stride(), source.storage_offset())
        copied.copy_(source)
        return copied

    # Warm the existing path; resets and exports are outside all performance loops.
    warm_state = state_copy() # Keep the temporary mutation target alive through synchronization.
    guarded_npu_call(lambda: normal(warm_state), torch.npu.synchronize)
    state_normal = state_copy()
    out_normal = guarded_npu_call(lambda: normal(state_normal), torch.npu.synchronize)
    state_profile = state_copy()
    timer = torch.zeros(timer_parser.TOTAL_BUFFER_SIZE, dtype=torch.int64, device=inputs["value"].device)

    def diagnostic():
        result = op(inputs["query"], inputs["key"], inputs["value"], state_profile, inputs["beta"],
                    inputs["actual_seq_lengths"], inputs["ssm_state_indices"], inputs["num_accepted_tokens"],
                    inputs["g"], inputs["gk"], timer, float(inputs["scale"]), _stable._current_stream_ptr())
        torch.autograd.graph.increment_version((state_profile, timer))
        return result

    out_profile = guarded_npu_call(diagnostic, torch.npu.synchronize)
    raw_path = output / "rank_0_npu_time.csv"
    # Save before validation/parsing so failed or zero captures remain available.
    with raw_path.open("w", encoding="utf-8") as file:
        file.writelines(f"{int(n)}\n" for n in timer.cpu().tolist())
    torch.save({"normal_out": out_normal.cpu(), "profile_out": out_profile.cpu(),
                "normal_state": state_normal.cpu(), "profile_state": state_profile.cpu()}, output / "outputs.pt")
    prefix = int(case.get("prefix_tokens", 0))
    if not torch.equal(out_normal[prefix:], out_profile[prefix:]) or not torch.equal(state_normal, state_profile):
        raise RuntimeError("diagnostic changed out or state; saved identical inputs and both outputs")
    metadata = parse_capture(raw_path, output, args.cycles_per_us, args.shared_clock)
    validate_capture_case(case, metadata)
    metadata.update({"case_spec": case, "normal_profile_bitwise_equal": True, "python": sys.executable,
                     "torch": torch.__version__, "package": str(package), "opapi": str(opapi),
                     "opapi_sha256": hashlib.sha256(opapi.read_bytes()).hexdigest(),
                     "stable_source_hash": actual_hash, "device_name": torch.npu.get_device_name(args.device)})
    write_json(output / "capture.json", metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case-id", type=int, default=0)
    parser.add_argument("--run-package", type=Path, help="exact .run produced by this build; mandatory online")
    parser.add_argument("--built-kernel", type=Path, help="override the current build's kernel .o root")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cycles-per-us", type=float, help="verified counter frequency; default cycles only")
    parser.add_argument("--shared-clock", action="store_true", help="assert relevant AIV counters have a common clock")
    parser.add_argument("--parse-raw", type=Path, help="offline parse only; does not import torch/ATK/CANN")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    timer_parser.validate_frequency(args.cycles_per_us)
    output = args.output or ROOT / "profiling_results/recurrent_gated_delta_rule" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = output.resolve()
    if args.worker:
        capture_worker(args, output)
        return
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "started", "case_id": args.case_id}
    write_json(output / "status.json", status)
    try:
        if args.parse_raw:
            parse_capture(args.parse_raw.resolve(), output, args.cycles_per_us, args.shared_clock)
            status["status"] = "offline_parse_only"
        else:
            if not args.run_package or not args.run_package.is_file():
                raise RuntimeError("--run-package must name this build's exact .run")
            check_python_environment()
            _, opapi = prepare_opp()
            built_root = (args.built_kernel or ROOT / "build/lib/fla_npu/opp/vendors/fla_npu_transformer/op_impl/ai_core/tbe/kernel").resolve()
            runtime_root = opapi.parents[2] / "op_impl/ai_core/tbe/kernel"
            write_json(output / "target_kernel_objects.json", verify_target_kernel_objects(runtime_root, built_root))
            check = run_logged([sys.executable, str(ROOT / "scripts/verify_libcust_opapi_md5.py"),
                                "--run-package", str(args.run_package.resolve()), "--built-kernel", str(built_root)],
                               output / "artifact_check.log")
            if ("[OK] loaded libcust_opapi.so matches the freshly built one." not in check or
                    not any("[OK]" in line and "kernel .o" in line and "match" in line for line in check.splitlines())):
                raise RuntimeError("artifact check did not prove current kernel .o was installed")
            case = load_spec(OP_DIR / "atk_recurrent_gated_delta_rule_perf.json", args.case_id)
            accuracy = json.loads((OP_DIR / "atk_recurrent_gated_delta_rule.json").read_text(encoding="utf-8"))
            matches = [index for index, c in enumerate(accuracy) if json.loads(next(x["range_values"] for x in c["inputs"]
                       if x["name"] == "case_spec")) == case]
            if not matches:
                raise RuntimeError("perf case has no identical original accuracy case; do not invent a new tolerance")
            performance = json.loads((OP_DIR / "atk_recurrent_gated_delta_rule_perf.json").read_text(encoding="utf-8"))
            perf_index = next(index for index, c in enumerate(performance) if c["id"] == args.case_id)
            env = os.environ.copy()
            # Parent is already in the activated CANN environment, with its installed
            # package's OPP prepared. Do not let a stale optional env script replace it.
            env["CANN_ENV"] = ""
            env["FLA_NPU_ENV"] = ""
            env["FLA_NPU_OPP_ENV"] = ""
            env["ATK_OUTPUT_ROOT"] = str(output / "atk_output")
            # ATK slices JSON list positions with an exclusive end, not case IDs.
            env["ACCURACY_START"] = str(matches[0])
            env["ACCURACY_END"] = str(matches[0] + 1)
            env["PERFORMANCE_START"] = str(perf_index)
            env["PERFORMANCE_END"] = str(perf_index + 1)
            base = ["bash", "tests/atk/run_test_cpu.sh", "-op=recurrent_gated_delta_rule",
                    f"-npu_device_id={args.device}", "-soc=ascend950"]
            run_logged(base + ["-scope=accuracy"], output / "atk_accuracy.log", env)
            verify_atk_stage(output / "atk_output/accuracy", "accuracy")
            run_logged(base + ["-scope=performance"], output / "atk_performance.log", env)
            verify_atk_stage(output / "atk_output/perf", "performance")
            worker = [sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(output),
                      "--device", str(args.device), "--case-id", str(args.case_id)]
            if args.cycles_per_us is not None:
                worker += ["--cycles-per-us", str(args.cycles_per_us)]
            if args.shared_clock:
                worker += ["--shared-clock"]
            run_logged(worker, output / "capture.log", timeout=300)
            write_json(output / "system_tasks.json", summarize_system_profile(output / "atk_output"))
            status["status"] = "captured_and_compared"
            status["precision_rule"] = "original ATK mixed_tolerance_bm plus normal/profile bitwise equality"
    except Exception as error:
        status.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "status.json", status)
        print(f"status={status['status']} output={output}", flush=True)


if __name__ == "__main__":
    main()

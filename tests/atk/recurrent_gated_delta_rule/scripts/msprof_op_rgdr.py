#!/usr/bin/env python3
"""Direct RGDR calls for `msprof op`, or offline OpBasicInfo statistics.

Uses the original ATK perf case and input builder. This is a diagnostic, not
ATK accuracy/performance acceptance. No torch imports in the offline path.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys


OP_REL = Path("tests/atk/recurrent_gated_delta_rule")
CASE_FILE = "atk_recurrent_gated_delta_rule_perf.json"
TASK_NAME = "aclnnRecurrentGatedDeltaRule_RecurrentGatedDeltaRule_RecurrentGatedDeltaRule"
KERNEL_NAME = re.compile(r"RecurrentGatedDeltaRule_[0-9a-f]{32}(?:_[0-9]+)?")
MISSING_TYPES = {"", "N/A", "NA", "UNKNOWN"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_case(root, case_id):
    cases = json.loads((Path(root) / OP_REL / CASE_FILE).read_text(encoding="utf-8"))
    matches = [case for case in cases if case["id"] == case_id]
    if len(matches) != 1:
        raise ValueError("case id must select exactly one original perf case")
    values = [item["range_values"] for item in matches[0]["inputs"] if item["name"] == "case_spec"]
    if len(values) != 1:
        raise ValueError("expected one original case_spec")
    spec = json.loads(values[0])
    expected = {"B": 1, "T": 1, "HK": 1, "HV": 1, "K": 128, "V": 128,
                "dtype": "bf16", "state_dtype": "bf16", "gate_mode": "g"}
    if any(spec.get(key) != value for key, value in expected.items()):
        raise ValueError("only the confirmed default A5 perf shape/dtypes/g-only case is supported")
    if (spec.get("repeat_calls", 1) != 1 or spec.get("accepted_tokens") is not None
            or spec.get("state_layout", "contiguous") != "contiguous"
            or spec.get("input_layout", "contiguous") != "contiguous"):
        raise ValueError("requires one call, contiguous inputs/state and no accepted-token input")
    return spec


def git_info(root):
    result = {}
    for key, args in (("commit", ["rev-parse", "HEAD"]),
                      ("branch", ["branch", "--show-current"]),
                      ("status", ["status", "--porcelain"])):
        try:
            run = subprocess.run(["git", *args], cwd=root, capture_output=True,
                                 text=True, encoding="utf-8", timeout=15)
            result[key] = run.stdout.strip() if run.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            result[key] = None
    return result


def run_application(args):
    root = args.repo_root.resolve()
    spec = load_case(root, args.case_id)
    # Prepare the installed package's OPP before torch_npu initializes the NPU.
    import fla_npu

    vendor = Path(fla_npu._prepare_embedded_opp())
    opapi = Path(fla_npu._resolve_packaged_opapi(vendor))

    import torch
    import torch_npu
    from fla_npu.ops import ascendc
    from fla_npu.ops.ascendc import _stable

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    device_name = torch.npu.get_device_name(args.device)
    if "950" not in device_name:
        raise RuntimeError("this diagnostic requires A5 / Ascend950")
    sys.path.insert(0, str(root / OP_REL))
    # Use build_inputs directly; do not call run_npu/FunctionApi, which also
    # regenerate inputs and run finite checks. ATK must be installed to import it.
    from executor_recurrent_gated_delta_rule import build_inputs

    objects = {str(path.relative_to(vendor)): sha256(path)
               for path in sorted(vendor.rglob("*.o"))
               if "recurrentgateddeltarule" in path.name.lower().replace("_", "")}
    report = {
        "status": "preparing", "label": args.label, "git": git_info(root),
        "case_id": args.case_id, "case_spec": spec, "calls": args.calls,
        "case_sha256": sha256(root / OP_REL / CASE_FILE),
        "executor_sha256": sha256(root / OP_REL / "executor_recurrent_gated_delta_rule.py"),
        "device": args.device, "device_name": device_name,
        "python": sys.executable, "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "package": str(Path(fla_npu.__file__).resolve()),
        "opapi": str(opapi), "opapi_sha256": sha256(opapi),
        "installed_rgdr_objects_sha256": objects,
        "artifact_scope": "installed package only; not a build/run-package equivalence check",
        "environment": {name: os.environ.get(name) for name in (
            "ASCEND_HOME_PATH", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_LAUNCH_BLOCKING",
            "TASK_QUEUE_ENABLE", "FLA_NPU_STABLE_ABI", "FLA_NPU_STABLE_STREAM", "FLA_NPU_STABLE_VALIDATE")},
        "measurement": "external msprof op; summarize OpBasicInfo Task Duration(us)",
        "warmup": "msprof op --warm-up; no extra RGDR warmup calls in Python",
        "accuracy": "not run; profiler may replay an in-place operator",
    }
    write_json(args.metadata, report)
    try:
        with torch.inference_mode():
            inputs = build_inputs(spec, device)
            # RGDR mutates state. Prepare all independent states BEFORE the first
            # RGDR call; no clone/reset/check is interleaved with target calls.
            states = [inputs["state"].clone() for _ in range(args.calls)]
            stream = torch.npu.current_stream()
            stream.synchronize()
            kwargs = {key: inputs[key] for key in (
                "beta", "scale", "actual_seq_lengths", "ssm_state_indices",
                "num_accepted_tokens", "g", "gk")}
            outputs = []  # Hold each output until its NPU work is complete.
            report["status"] = "calling"
            write_json(args.metadata, report)
            for state in states:
                outputs.append(ascendc.recurrent_gated_delta_rule(
                    inputs["query"], inputs["key"], inputs["value"], state, **kwargs))
                stream.synchronize()
            report.update(status="application_complete", completed_calls=len(outputs),
                          stream=int(stream.npu_stream),
                          public_function_module=ascendc.recurrent_gated_delta_rule.__module__,
                          stable=_stable._loaded_path,
                          stable_sha256=sha256(_stable._loaded_path) if _stable._loaded_path else None)
    except Exception as error:
        report.update(status="application_failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.metadata, report)
    print(f"RGDR application completed {args.calls} calls; metadata={args.metadata}")
    print("Timing comes from msprof op, not Python wall clock. Run --summarize after msprof exits.")


def rgdr_match_source(row):
    kind = (row.get("Op Type", row.get("OP Type", "")) or "").strip()
    name = (row.get("Op Name") or "").strip()
    if kind.lower().replace("_", "") == "recurrentgateddeltarule":
        return "op_type"
    # Real A5 msprof op export: Op Type=vector describes the compute kind,
    # not the semantic operator type. Require the complete generated RGDR
    # kernel name; never count other vector operators or suffix/substring hits.
    if kind.lower() == "vector" and KERNEL_NAME.fullmatch(name):
        return "vector_kernel_name"
    if kind.upper() in MISSING_TYPES and (
            name in {"RecurrentGatedDeltaRule", "recurrent_gated_delta_rule", TASK_NAME}
            or KERNEL_NAME.fullmatch(name)):
        return "exact_op_name"
    return None


def is_rgdr(row):
    return rgdr_match_source(row) is not None


def statistics_us(values):
    ordered = sorted(values)

    def percentile(q):
        index = (len(ordered) - 1) * q
        low, high = math.floor(index), math.ceil(index)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)

    mean = statistics.mean(values)
    deviation = statistics.pstdev(values)
    return {"count": len(values), "mean_us": mean, "median_us": statistics.median(values),
            "std_us": deviation, "cv_pct": deviation / mean * 100,
            "min_us": min(values), "max_us": max(values),
            "p50_us": percentile(.5), "p90_us": percentile(.9), "p99_us": percentile(.99)}


def summarize_msprof(directory, expected_count, output):
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError(f"profile directory does not exist: {root}")
    files = sorted(path for path in root.rglob("OpBasicInfo*.csv") if "dump" not in path.relative_to(root).parts)
    report = {"status": "failed", "profile_root": str(root), "expected_count": expected_count,
              "metric": "msprof op / OpBasicInfo.csv / Task Duration(us)", "inventory": [], "samples": []}
    try:
        if not files:
            report["csv_files"] = [str(path.relative_to(root)) for path in sorted(root.rglob("*.csv"))]
            raise ValueError("no OpBasicInfo*.csv; inspect CSV inventory in the output report")
        runs = {next((parent for parent in path.parents if parent.name.startswith("OPPROF_")), root)
                for path in files}
        if len(runs) != 1:
            raise ValueError("multiple OPPROF runs found; summarize one fresh run directory at a time")
        for path in files:
            with path.open(encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                entry = {"file": str(path), "columns": reader.fieldnames, "rgdr_count": 0,
                         "row_count": 0, "unmatched_row_examples": [], "rgdr_match_sources": {}}
                report["inventory"].append(entry)
                if not reader.fieldnames or "Task Duration(us)" not in reader.fieldnames or "Op Name" not in reader.fieldnames:
                    raise ValueError(f"unsupported OpBasicInfo schema/units: {path}")
                kinds = Counter()
                for line, row in enumerate(reader, start=2):
                    entry["row_count"] += 1
                    kinds[row.get("Op Type", row.get("OP Type", ""))] += 1
                    match_source = rgdr_match_source(row)
                    if match_source is None:
                        if len(entry["unmatched_row_examples"]) < 3:
                            entry["unmatched_row_examples"].append({"line": line, "row": row})
                        continue
                    sources = entry["rgdr_match_sources"]
                    sources[match_source] = sources.get(match_source, 0) + 1
                    try:
                        duration = float(row["Task Duration(us)"])
                    except (TypeError, ValueError) as error:
                        raise ValueError(f"invalid Task Duration(us): {path}:{line}") from error
                    if not math.isfinite(duration) or duration <= 0:
                        raise ValueError(f"nonpositive/nonfinite Task Duration(us): {path}:{line}")
                    report["samples"].append({"file": str(path), "line": line, "duration_us": duration,
                                              "op_name": row["Op Name"], "match_source": match_source, "row": row})
                    entry["rgdr_count"] += 1
                entry["type_counts"] = dict(kinds)
        count = len(report["samples"])
        if count != expected_count:
            raise ValueError(f"expected {expected_count} RGDR rows, got {count}; "
                             f"inspect row_count/type_counts/unmatched_row_examples in {output}; "
                             "do not pool partial/duplicate runs")
        devices = {(sample["row"].get(column) or "").strip() for sample in report["samples"]
                   for column in ("Device ID", "Device Id")} - MISSING_TYPES
        if len(devices) > 1:
            raise ValueError("multiple devices found; summarize one device at a time")
        report["devices"] = sorted(devices)
        report["statistics"] = statistics_us([sample["duration_us"] for sample in report["samples"]])
        report["status"] = "complete"
        return report
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output, report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case-id", type=int, default=0)
    parser.add_argument("--calls", type=int, default=20, help="RGDR calls; match msprof --launch-count")
    parser.add_argument("--label", default="rgdr")
    parser.add_argument("--metadata", type=Path, help="application provenance JSON (not timing statistics)")
    parser.add_argument("--summarize", type=Path, help="offline: one msprof op output directory")
    parser.add_argument("--expected-count", type=int, default=20, help="required RGDR rows for --summarize")
    parser.add_argument("--output", type=Path, help="offline summary JSON; default PROFILE/rgdr_task_summary.json")
    args = parser.parse_args(argv)
    if not 1 <= args.calls <= 100 or args.expected_count < 1 or args.device < 0:
        parser.error("calls must be in [1, 100], expected-count positive, and device nonnegative")
    if args.summarize:
        output = args.output or args.summarize / "rgdr_task_summary.json"
        report = summarize_msprof(args.summarize, args.expected_count, output)
        stats = report["statistics"]
        print("RGDR Task Duration(us): " + " ".join(
            f"{key}={value:.5f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in stats.items()))
        print(f"summary={output}")
    else:
        if sys.platform != "linux":
            parser.error("online execution requires Linux + A5; --summarize works offline")
        if args.output:
            parser.error("--output is for --summarize; use --metadata for application provenance")
        if args.metadata is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            args.metadata = args.repo_root / OP_REL / "runs/msprof_op" / stamp / "application.json"
        run_application(args)


if __name__ == "__main__":
    main()

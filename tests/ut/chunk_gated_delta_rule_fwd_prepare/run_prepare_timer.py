"""Diagnostic raw-ACLNN launcher for Prepare AscendTimerV2.

The public Stable ABI is unchanged. This test owns the ACLNN workspace and
reads the timer tail after the same Prepare kernel has completed. Standalone
mode reuses the original Prepare ATK case JSON and input generator. Fused c0/c1
callers can import ``run_timer`` and pass the exact tensors from their original
input generator.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_TEST_DIR = Path(__file__).resolve().parent
_TESTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_TESTS_DIR / "atk" / "common"))
sys.path.insert(0, str(_TEST_DIR))

from aclnn_error_probe import _CallContext, runtime  # noqa: E402
import parse_timer_csv as timer_parser  # noqa: E402
import trace_parser  # noqa: E402

_SYMBOL = "aclnnChunkGatedDeltaRuleFwdPrepare"


def _host_ints(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [int(v) for v in value.detach().cpu().reshape(-1).tolist()]
    return [int(v) for v in value]


def _canonical_chunk_indices(cu_seqlens, chunk_size):
    values = _host_ints(cu_seqlens)
    if values is None:
        return None
    pairs = []
    for sequence, (begin, end) in enumerate(zip(values, values[1:])):
        for local_chunk in range((end - begin + chunk_size - 1) // chunk_size):
            pairs.extend((sequence, local_chunk))
    return pairs


def _allocate_outputs(inputs):
    q, k, v, g = (inputs[name] for name in ("q", "k", "v", "g"))
    batch, key_heads, tokens, key_dim = q.shape
    value_heads = v.shape[1]
    chunk_size = int(inputs["chunk_size"])
    use_l2 = bool(inputs["use_qk_l2norm_in_kernel"])
    use_beta = bool(inputs["use_beta_sigmoid_in_kernel"])
    options = {"device": q.device}
    return {
        "g_out": torch.empty((batch, value_heads, tokens), dtype=torch.float32, **options),
        "w": torch.empty((batch, value_heads, tokens, key_dim), dtype=q.dtype, **options),
        "u": torch.empty_like(v),
        "a": torch.empty((batch, value_heads, tokens, chunk_size), dtype=q.dtype, **options),
        "q_hat": torch.empty_like(q) if use_l2 else None,
        "k_hat": torch.empty_like(k) if use_l2 else None,
        "q_rstd": torch.empty((batch, key_heads, tokens), dtype=torch.float32, **options) if use_l2 else None,
        "k_rstd": torch.empty((batch, key_heads, tokens), dtype=torch.float32, **options) if use_l2 else None,
        "beta_eff": torch.empty((batch, value_heads, tokens), dtype=torch.float32, **options) if use_beta else None,
    }


def _check_status(api, status, phase):
    if status == 0:
        return
    raise RuntimeError(f"{_SYMBOL} {phase} failed: aclnnStatus={status}: {api._recent_error_message()}")


def launch_raw(inputs):
    """Launch Prepare once and return ``(outputs, timer_int64_numpy)``."""
    api = runtime()
    context = _CallContext(api)
    executor = ctypes.c_void_p()
    workspace_size = ctypes.c_uint64()
    outputs = _allocate_outputs(inputs)
    try:
        tensor_names = ("q", "k", "v", "g", "beta")
        tensors = [context.tensor(inputs[name], name, acl_format_override=2) for name in tensor_names]
        use_gate = bool(inputs["use_gate_in_kernel"])
        for name in ("a_log", "dt_bias"):
            value = inputs.get(name) if use_gate else None
            tensors.append(context.tensor(value, name, acl_format_override=2))
        cu_values = _host_ints(inputs.get("cu_seqlens"))
        chunk_values = _host_ints(inputs.get("chunk_indices"))
        if cu_values is not None and chunk_values is None:
            chunk_values = _canonical_chunk_indices(cu_values, int(inputs["chunk_size"]))
        tensors.append(context.int_array(cu_values))
        tensors.append(context.int_array(chunk_values))
        output_names = ("g_out", "w", "u", "a", "q_hat", "k_hat", "q_rstd", "k_rstd", "beta_eff")
        output_handles = [context.tensor(outputs[name], name, acl_format_override=2) for name in output_names]

        get_workspace = api.symbol(_SYMBOL + "GetWorkspaceSize")
        get_workspace.argtypes = (
            [ctypes.c_void_p] * 9
            + [ctypes.c_int64, ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
            + [ctypes.c_void_p] * 9
            + [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)]
        )
        get_workspace.restype = ctypes.c_int
        api._recent_error_message()
        status = get_workspace(
            *tensors,
            int(inputs["chunk_size"]),
            bool(inputs["allow_neg_eigval"]),
            bool(inputs["use_exp2"]),
            bool(inputs.get("output_a", True)),
            *output_handles,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        _check_status(api, status, "GetWorkspaceSize")
        if not executor.value or workspace_size.value < timer_parser.TOTAL_BUFFER_SIZE * 8:
            raise RuntimeError(
                "diagnostic workspace is missing the timer tail; rebuild with PREPARE_TIMER_DIAGNOSTIC=1"
            )

        workspace = torch.zeros(workspace_size.value, dtype=torch.uint8, device=inputs["q"].device)
        launch = api.symbol(_SYMBOL)
        launch.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p]
        launch.restype = ctypes.c_int
        stream = ctypes.c_void_p(int(torch.npu.current_stream().npu_stream))
        api._recent_error_message()
        status = launch(
            ctypes.c_void_p(workspace.data_ptr()),
            workspace_size,
            executor,
            stream,
        )
        _check_status(api, status, "launch")
        torch.npu.synchronize()

        timer_bytes = timer_parser.TOTAL_BUFFER_SIZE * 8
        timer_u8 = workspace[workspace_size.value - timer_bytes:].cpu().numpy().tobytes()
        timer = np.frombuffer(timer_u8, dtype=np.int64).copy()
        if timer.size != timer_parser.TOTAL_BUFFER_SIZE:
            raise RuntimeError(f"timer tail has {timer.size} int64 values, expected {timer_parser.TOTAL_BUFFER_SIZE}")
        ordered_outputs = (
            outputs["q_hat"] if outputs["q_hat"] is not None else inputs["q"],
            outputs["k_hat"] if outputs["k_hat"] is not None else inputs["k"],
            outputs["q_rstd"],
            outputs["k_rstd"],
            outputs["beta_eff"] if outputs["beta_eff"] is not None else inputs["beta"].float(),
            outputs["g_out"],
            outputs["w"],
            outputs["u"],
            outputs["a"],
        )
        return ordered_outputs, timer
    finally:
        if executor.value:
            destroy = api.symbol("aclDestroyOpExecutor")
            destroy.argtypes, destroy.restype = [ctypes.c_void_p], ctypes.c_int
            destroy(executor)
        context.destroy()


def _public_call(inputs):
    from fla_npu.ops import ascendc

    return ascendc.chunk_gated_delta_rule_fwd_prepare(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"],
        chunk_size=int(inputs["chunk_size"]),
        use_qk_l2norm_in_kernel=bool(inputs["use_qk_l2norm_in_kernel"]),
        use_gate_in_kernel=bool(inputs["use_gate_in_kernel"]),
        use_beta_sigmoid_in_kernel=bool(inputs["use_beta_sigmoid_in_kernel"]),
        allow_neg_eigval=bool(inputs["allow_neg_eigval"]),
        use_exp2=bool(inputs["use_exp2"]),
        output_a=bool(inputs.get("output_a", True)),
        a_log=inputs.get("a_log"),
        dt_bias=inputs.get("dt_bias"),
        cu_seqlens=inputs.get("cu_seqlens"),
        chunk_indices=inputs.get("chunk_indices"),
    )


def _assert_same_outputs(reference, actual, *, compare_a=True):
    if len(reference) != len(actual):
        raise AssertionError("public/raw output count differs")
    for index, (expected, value) in enumerate(zip(reference, actual)):
        # output_a=False leaves separately allocated A buffers unwritten.
        if index == 8 and not compare_a:
            continue
        if expected is None or value is None:
            if (expected is None) != (value is None):
                raise AssertionError(f"output {index}: optional presence differs")
            continue
        torch.testing.assert_close(value, expected, rtol=0, atol=0, equal_nan=True)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_timer(inputs, output_dir, *, compare_public=True, cycles_per_us=None, shared_clock=False):
    """Run the raw diagnostic path and export raw/CSV/metadata outside timing loops."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if compare_public:
        expected = _public_call(inputs)
        torch.npu.synchronize()
    actual, timer = launch_raw(inputs)

    # Preserve the diagnostic evidence even if instrumentation changed output.
    raw_path = output_dir / "rank_0_npu_time.csv"
    parsed_path = output_dir / "rank_0_npu_time_parsed.csv"
    np.savetxt(raw_path, timer.reshape(-1, 1), fmt="%d")
    if compare_public:
        _assert_same_outputs(expected, actual, compare_a=bool(inputs.get("output_a", True)))

    timer_parser.parse_and_dump(str(raw_path), str(parsed_path), cycles_per_us=cycles_per_us)
    timer_parser.print_summary(str(parsed_path))

    trace_path = None
    if cycles_per_us is not None:
        if not shared_clock:
            raise ValueError("trace export requires explicit shared_clock=True")
        base_cycle = trace_parser.find_min_start(parsed_path)
        events = trace_parser.generate_trace_json(
            parsed_path, 0, base_cycle, cycles_per_us=cycles_per_us, shared_clock=True
        )
        trace_path = output_dir / "rank_0_npu_time.json"
        with open(trace_path, "w", encoding="utf-8") as stream:
            json.dump(events, stream, indent=2, ensure_ascii=False, allow_nan=False)

    metadata = {
        "symbol": _SYMBOL,
        "timer_layout": "workspace_tail_start_end_int64",
        "timer_int64_values": int(timer.size),
        "raw_sha256": _sha256(raw_path),
        "parsed_sha256": _sha256(parsed_path),
        "trace_sha256": _sha256(trace_path) if trace_path is not None else None,
        "public_raw_bitwise_match": bool(compare_public),
        "output_a_compared": bool(compare_public and inputs.get("output_a", True)),
        "cycles_per_us": cycles_per_us,
        "shared_clock": bool(shared_clock),
        "kernel_window_semantics": "per-core instruction-stream endpoint; no added barrier",
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
    return {"raw": raw_path, "parsed": parsed_path, "trace": trace_path, "metadata": metadata_path}


def _load_atk_case(case_index):
    case_path = _TESTS_DIR / "atk" / "chunk_gated_delta_rule_fwd_prepare" / "atk_chunk_gated_delta_rule_fwd_prepare_perf.json"
    with open(case_path, encoding="utf-8") as stream:
        cases = json.load(stream)
    entry = cases[case_index]
    case_spec = next(item for item in entry["inputs"] if item["name"] == "case_spec")
    return json.loads(case_spec["range_values"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, choices=(0, 1), default=0, help="Prepare ATK perf case index")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-public-compare", action="store_true")
    parser.add_argument("--cycles-per-us", type=float)
    parser.add_argument("--shared-clock", action="store_true")
    args = parser.parse_args()

    import torch_npu  # noqa: F401

    os.environ["TBE_PARALLEL_COMPILE_ENABLE"] = "0"
    os.environ["PARALLEL_COMPILE"] = "0"
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(args.device)
    spec = _load_atk_case(args.case)
    prepare_dir = _TESTS_DIR / "atk" / "chunk_gated_delta_rule_fwd_prepare"
    sys.path.insert(0, str(prepare_dir))
    from executor_chunk_gated_delta_rule_fwd_prepare import build_inputs

    inputs = build_inputs(spec, torch.device(f"npu:{args.device}"), high_precision=False)
    paths = run_timer(
        inputs,
        args.output_dir,
        compare_public=not args.skip_public_compare,
        cycles_per_us=args.cycles_per_us,
        shared_clock=args.shared_clock,
    )
    with open(args.output_dir / "case_spec.json", "w", encoding="utf-8") as stream:
        json.dump(spec, stream, indent=2, ensure_ascii=False)
    print(json.dumps({key: str(value) if value is not None else None for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()

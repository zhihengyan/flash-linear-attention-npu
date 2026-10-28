"""CPU-only layout/parser integrity tests; these do not simulate device timing."""
import contextlib
import io
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import parse_timer_csv as p
import profile_performance as collector
import trace_parser

ROOT = Path(__file__).resolve().parents[4]
KERNEL = ROOT / "fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/op_kernel"


def valid_raw():
    raw = [0] * p.TOTAL_BUFFER_SIZE
    raw[p.HEADER_OFFSET:] = [p.RGDR_TIMER_MAGIC, 1, p.TOTAL_BUFFER_SIZE, 1,
                            1, 1, 1, 1, 128, 2, 2, 2,
                            p.FIXED_TIMING_COUNT, p.DYNAMIC_TYPE_COUNT, p.MAX_DYNAMIC_ITER, 1]
    raw[0:2] = [100, 1000]
    raw[2:8] = [110, 150, 160, 170, 155, 900]
    raw[p.META_OFFSET + 32] = 0
    raw[p.META_OFFSET + 33] = 1
    raw[p.META_OFFSET + 63] = p.RGDR_TIMER_MAGIC
    raw[p.META_OFFSET] = 1
    raw[p.N_TIMING_COUNTER] = 1
    offset = p.FIXED_TIMING_COUNT * 2
    raw[offset:offset + 2] = [200, 300]
    return raw


class TimerLayoutTest(unittest.TestCase):
    def test_dfx_matches_enclosing_api(self):
        # CANN rejects L2 DFX in helpers whose name differs from the API name.
        source = (KERNEL.parent / "op_host/op_api/aclnn_recurrent_gated_delta_rule.cpp").read_text(encoding="utf-8")
        declarations = list(re.finditer(r"\baclnnStatus\s+(\w+)\s*\([^)]*\)\s*\{", source))
        observed = set()
        for index, function in enumerate(declarations):
            end = declarations[index + 1].start() if index + 1 < len(declarations) else len(source)
            body = source[function.end():end]
            for macro in re.finditer(r"\bL2_DFX_PHASE_([12])\s*\(\s*(\w+)\s*[,)]", body):
                phase, api = macro.groups()
                expected = api + ("GetWorkspaceSize" if phase == "1" else "")
                self.assertEqual(function.group(1), expected, "Invalid DFX in " + function.group(1))
                observed.add((api, phase))
        self.assertEqual(observed, {(api, phase)
                                   for api in ("aclnnRecurrentGatedDeltaRule", "aclnnRecurrentGatedDeltaRuleProfile")
                                   for phase in ("1", "2")})

    def test_cpp_enum_and_constants(self):
        common = (KERNEL / "AscendTimerV2.hpp").read_text(encoding="utf-8")
        for enum, sentinel, expected in (
            ("FixedTiming", "FIXED_TIMING_COUNT", ["KERNEL_TIMING_IDX"] + list(p.FIXED_TIMING_NAMES.values())[1:]),
            ("DynamicTimingType", "DYNAMIC_TYPE_COUNT_ENUM", list(p.DYNAMIC_TIMING_NAMES.values())),
        ):
            block = re.search(r"enum " + enum + r"\s*:\s*int32_t\s*\{(.*?)\}", common, re.S).group(1)
            names = [x.strip() for x in block.split(",") if x.strip()]
            self.assertEqual(names, expected + [sentinel])
        for name in ("MAX_DYNAMIC_ITER", "N_CORE_COUNT", "META_WORDS_PER_CORE", "HEADER_WORDS"):
            self.assertEqual(int(re.search(name + r"\s*=\s*(\d+)", common).group(1)), getattr(p, name))
        self.assertEqual(p.META_OFFSET % 16, 0)
        self.assertEqual(p.HEADER_OFFSET % 16, 0)

    def test_all_dynamic_pairs_match_enum(self):
        source = (KERNEL / "arch35/recurrent_gated_delta_rule.h").read_text(encoding="utf-8")
        begin = re.findall(r"RGDR_TIMER_BEGIN\((\w+), (\w+)\);", source)
        end = re.findall(r"RGDR_TIMER_END\((\w+), (\w+)\);", source)
        self.assertCountEqual(begin, end)
        self.assertEqual(len(begin), len(p.DYNAMIC_TIMING_NAMES))
        self.assertEqual({kind for _, kind in begin}, set(p.DYNAMIC_TIMING_NAMES.values()))

    def test_valid_linear_aiv_metadata(self):
        self.assertEqual(p.get_core_info(127), (127, "AIV", 0))
        self.assertEqual(p.get_base_index(127), 127 * p.N_TIMING_COUNTER_PER_CORE_ALIGN)
        self.assertEqual(p.validate_metadata(valid_raw())["block_dim"], 1)

    def test_reject_zero_wrong_capacity_missing_pair(self):
        for change in ("zero", "capacity", "pair", "core", "count", "fixed"):
            raw = valid_raw()
            if change == "zero": raw = [0] * len(raw)
            if change == "capacity": raw.pop()
            if change == "pair": raw[p.FIXED_TIMING_COUNT * 2 + 1] = 0
            if change == "core": raw[p.META_OFFSET + 63] = 0
            if change == "count": raw[p.N_TIMING_COUNTER] = 2
            if change == "fixed": raw[2:4] = [0, 0]
            with self.subTest(change=change), self.assertRaises(ValueError):
                p.validate_metadata(raw)

    def test_truncation_preserves_true_count(self):
        raw = valid_raw()
        raw[p.META_OFFSET] = 23
        raw[p.N_TIMING_COUNTER] = p.MAX_DYNAMIC_ITER
        for slot in range(p.MAX_DYNAMIC_ITER):
            offset = (p.FIXED_TIMING_COUNT + slot) * 2
            raw[offset:offset + 2] = [200 + slot, 300 + slot]
        self.assertTrue(p.validate_metadata(raw)["truncated"])

    def test_raw_parsing_and_trace_frequency_guard(self):
        parent = ROOT / "profiling_results"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            path = Path(temp).resolve()
            path.relative_to(parent.resolve()) # Verify the automatic recursive cleanup stays in the workspace.
            raw = path / "raw.csv"
            raw.write_text("\n".join(map(str, valid_raw())) + "\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                result = collector.parse_capture(raw, path)
            self.assertIn("not generated", result["trace"])
            parsed = path / "rank_0_npu_time_parsed.csv"
            with self.assertRaises(ValueError):
                trace_parser.generate_trace_json(parsed, 0, 200)
            events = trace_parser.generate_trace_json(parsed, 0, trace_parser.find_min_start(parsed), 1000, True)
            self.assertEqual(events[0]["args"]["core_type"], "AIV")
            self.assertEqual(next(x for x in events if x["name"] == "BATCH_GATES_ISSUE_0")["dur"], 0.1)
            json.dumps(events, allow_nan=False)

    def test_original_perf_case_is_accuracy_case(self):
        directory = ROOT / "tests/atk/recurrent_gated_delta_rule"
        self.assertEqual(collector.load_spec(directory / "atk_recurrent_gated_delta_rule_perf.json", 0),
                         collector.load_spec(directory / "atk_recurrent_gated_delta_rule.json", 0))

    def test_default_case_rejects_missing_actual_work(self):
        case = collector.load_spec(ROOT / "tests/atk/recurrent_gated_delta_rule/atk_recurrent_gated_delta_rule_perf.json", 0)
        metadata = p.validate_metadata(valid_raw())
        with self.assertRaisesRegex(ValueError, "actual timer phase counts"):
            collector.validate_capture_case(case, metadata)
        metadata["actual_calls_per_core"] = [{name: 0 if name in ("GK_COPY_WAIT", "GK_EXP_ISSUE") else 1
                                              for name in p.DYNAMIC_TIMING_NAMES.values()}]
        collector.validate_capture_case(case, metadata)

    def test_private_adapter_coverage_guard(self):
        file = ROOT / "torch_custom/fla_npu/tools/stable_coverage.py"
        spec = importlib.util.spec_from_file_location("rgdr_coverage_test", file)
        coverage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(coverage)
        name = "_profile_recurrent_gated_delta_rule"
        self.assertNotIn(name, coverage.public_ops())
        original = coverage.adapters()
        self.assertTrue(original[name]["impl"])
        broken = {key: dict(value) for key, value in original.items()}
        broken[name]["run"] = None
        with patch.object(coverage, "adapters", return_value=broken):
            self.assertTrue(any("private diagnostic wiring is incomplete" in x for x in coverage.evaluate()["blockers"]))

    def test_target_kernel_objects_must_exist_and_match(self):
        parent = ROOT / "profiling_results"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            directory = Path(temp).resolve()
            directory.relative_to(parent.resolve())
            runtime, built = directory / "runtime", directory / "built"
            runtime.mkdir()
            built.mkdir()
            with self.assertRaises(ValueError):
                collector.verify_target_kernel_objects(runtime, built)
            for root in (runtime, built):
                target = root / "ascend950/recurrent_gated_delta_rule/kernel.o"
                target.parent.mkdir(parents=True)
                target.write_bytes(b"test object bytes")
            self.assertEqual(len(collector.verify_target_kernel_objects(runtime, built)["sha256"]), 1)
            (built / "ascend950/recurrent_gated_delta_rule/kernel.o").write_bytes(b"different object")
            with self.assertRaises(ValueError):
                collector.verify_target_kernel_objects(runtime, built)


if __name__ == "__main__":
    unittest.main()

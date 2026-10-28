"""CPU-only layout/parser integrity tests; these do not simulate device timing."""
import contextlib
import io
import importlib.util
import json
from pathlib import Path
import re
import sys
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
    def test_single_case_atk_ranges_select_list_positions(self):
        parent = ROOT / "profiling_results"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            directory = Path(temp).resolve()
            directory.relative_to(parent.resolve())
            source = json.loads((collector.OP_DIR / "atk_recurrent_gated_delta_rule.json").read_text(encoding="utf-8"))
            accuracy = [dict(source[1], id=1000), dict(source[0], id=900)]
            performance = [dict(source[0], id=700)]
            for suffix, cases in (("", accuracy), ("_perf", performance)):
                (directory / f"atk_recurrent_gated_delta_rule{suffix}.json").write_text(json.dumps(cases), encoding="utf-8")
            run = directory / "test.run"
            run.touch()
            output = directory / "capture"
            observed = []

            def fake_run(command, log, env=None, timeout=None):
                if command[0] == "bash":
                    stage = "accuracy" if "-scope=accuracy" in command else "performance"
                    prefix = stage.upper()
                    cases = accuracy if stage == "accuracy" else performance
                    selected = cases[int(env[prefix + "_START"]):int(env[prefix + "_END"])]
                    self.assertEqual(len(selected), 1, "ATK must receive a nonempty single-case range")
                    self.assertEqual(selected[0]["id"], 900 if stage == "accuracy" else 700)
                    observed.append(stage)
                    nested = output / "atk_output" / ("accuracy" if stage == "accuracy" else "perf") / "task/log/atk.log"
                    nested.parent.mkdir(parents=True)
                    nested.write_text("Total Task: 1, success 1, failed 0\nacc_pass_result:Pass\n", encoding="utf-8")
                return ("[OK] loaded libcust_opapi.so matches the freshly built one.\n"
                        "[OK] all 2 kernel .o files match the freshly built OPP.")

            with patch.object(sys, "argv", ["profile_performance.py", "--case-id", "700",
                                             "--run-package", str(run), "--output", str(output)]), \
                    patch.object(collector, "OP_DIR", directory), \
                    patch.object(collector, "check_python_environment"), \
                    patch.object(collector, "prepare_opp", return_value=(directory, directory / "opp/op_api/lib/libcust_opapi.so")), \
                    patch.object(collector, "verify_target_kernel_objects", return_value={}), \
                    patch.object(collector, "run_logged", side_effect=fake_run), \
                    patch.object(collector, "summarize_system_profile", return_value=[]), \
                    contextlib.redirect_stdout(io.StringIO()):
                collector.main()
            self.assertEqual(observed, ["accuracy", "performance"])

    def test_atk_stage_rejects_failed_empty_or_missing_reports(self):
        parent = ROOT / "profiling_results"
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            directory = Path(temp).resolve()
            directory.relative_to(parent.resolve())
            with self.assertRaisesRegex(RuntimeError, "no task-count evidence"):
                collector.verify_atk_stage(directory, "accuracy")
            log = directory / "task/log/atk.log"
            log.parent.mkdir(parents=True)
            for content in ("atk task failed!\nTotal Task: 0, success 0, failed 0",
                            "Total Task: 0, success 0, failed 0",
                            "Total Task: 1, success 0, failed 1",
                            "Total Task: 2, success 2, failed 0",
                            "no summary was generated"):
                log.write_text(content, encoding="utf-8")
                for stage in ("accuracy", "performance"):
                    with self.subTest(stage=stage, content=content), self.assertRaises(RuntimeError):
                        collector.verify_atk_stage(directory, stage)
            for flags in ("", "Failed", "None", "Pass\nacc_pass_result:Failed", "Failed\nacc_pass_result:Pass"):
                log.write_text("Total Task: 1, success 1, failed 0\nacc_pass_result:" + flags, encoding="utf-8")
                with self.subTest(flags=flags), self.assertRaisesRegex(RuntimeError, "requires acc_pass_result"):
                    collector.verify_atk_stage(directory, "accuracy")
            self.assertEqual(collector.verify_atk_stage(directory, "performance"), [str(log)])
            log.write_text("Total Task: 1, success 1, failed 0\nacc_pass_result: Pass", encoding="utf-8")
            self.assertEqual(collector.verify_atk_stage(directory, "accuracy"), [str(log)])

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

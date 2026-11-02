"""Offline control/evidence tests; these do not simulate NPU execution."""
from contextlib import nullcontext
import csv
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import measure_public_latency as bench

ROOT = Path(__file__).resolve().parents[4]
REPORTED_TASK_NAME = "aclnnRecurrentGatedDeltaRule_RecurrentGatedDeltaRule_RecurrentGatedDeltaRule"


class TimingControlTest(unittest.TestCase):
    def test_wall_window_excludes_reset_and_holds_output_until_sync(self):
        log, state, held = [], {"value": 99}, []
        ticks = iter((100_000, 103_000, 110_000))

        class Output:
            pass

        def prepare():
            log.extend(("restore", "prepare_sync"))
            state["value"] = 7

        def clock():
            log.append("clock")
            return next(ticks)

        def call():
            self.assertEqual(state["value"], 7)
            state["value"] = 8
            result = Output()
            held.append(weakref.ref(result))
            log.append("call")
            return result

        def sync():
            self.assertIsNotNone(held[-1]())
            log.append("completion_sync")

        output, values = bench.measure_wall(call, prepare, sync, clock)
        self.assertIs(held[-1](), output)
        self.assertEqual(log, ["restore", "prepare_sync", "clock", "call", "clock", "completion_sync", "clock"])
        self.assertEqual(values, {"host_submit_us": 3, "call_and_sync_us": 10, "post_return_wait_us": 7})

    def test_each_sample_restores_initial_state(self):
        state, seen = [0], []

        def prepare():
            state[0] = 10

        def call():
            seen.append(state[0])
            state[0] += 1
            return state[0]

        for _ in range(3):
            ticks = iter((0, 1000, 2000))
            bench.measure_wall(call, prepare, lambda: None, lambda: next(ticks))
        self.assertEqual(seen, [10, 10, 10])

    def test_events_use_same_stream_and_read_elapsed_after_completion(self):
        log = []
        stream, output = object(), object()

        class Event:
            def __init__(self, name):
                self.name = name

            def record(self, actual):
                self_outer.assertIs(actual, stream)
                log.append(self.name)

            def elapsed_time(self, other):
                self_outer.assertEqual(log[-1], "sync")
                self_outer.assertEqual(other.name, "end")
                return .0045  # API unit is milliseconds.

        self_outer = self
        start, end = Event("start"), Event("end")
        def call():
            log.append("call")
            return output
        actual, values = bench.measure_event(call, lambda: log.append("prepare"), lambda: log.append("sync"),
                                            start, end, stream)
        self.assertIs(actual, output)
        self.assertEqual(log, ["prepare", "start", "call", "end", "sync"])
        self.assertAlmostEqual(values["stream_event_us"], 4.5)

    def test_profile_states_are_prepared_before_capture(self):
        active, clones, submitted = [False], [], []

        class State:
            def clone(self):
                self_outer.assertFalse(active[0])
                result = object()
                clones.append(result)
                return result

        class Capture:
            step_number = 0

            def __enter__(self):
                self_outer.assertEqual(len(clones), 4)
                active[0] = False  # First step is profiler warmup.
                return self

            def __exit__(self, *args):
                active[0] = False

            def step(self):
                self.step_number += 1
                active[0] = 1 <= self.step_number <= 3

        self_outer = self
        def create_profile(**kwargs):
            self.assertFalse(kwargs["experimental_config"]["data_simplification"])
            self.assertEqual(kwargs["experimental_config"]["aic_metrics"], "none")
            self.assertEqual(kwargs["schedule"], {"wait": 0, "warmup": 1, "active": 3, "repeat": 1})
            return Capture()
        profiler = SimpleNamespace(
            ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
            AiCMetrics=SimpleNamespace(AiCoreNone="none"), schedule=lambda **kw: kw,
            ProfilerLevel=SimpleNamespace(Level0=0), _ExperimentalConfig=lambda **kw: kw,
            tensorboard_trace_handler=lambda path: path, profile=create_profile)
        torch = SimpleNamespace(profiler=SimpleNamespace(record_function=lambda name: nullcontext()))
        recorded = []
        def call(state):
            recorded.append(active[0])
            self.assertIn(state, clones)
            submitted.append(state)
            return state
        with patch.object(bench, "npu_deadline", nullcontext):
            outputs, states = bench.profile_calls(torch, SimpleNamespace(profiler=profiler), call, State(),
                                                  lambda: None, 3, Path("unused-profile-path"))
        self.assertEqual(states, clones)
        self.assertEqual(outputs, submitted)
        self.assertEqual(len({id(s) for s in submitted}), 4)
        self.assertEqual(recorded, [False, True, True, True])


class EvidenceTest(unittest.TestCase):
    def test_case_reuses_original_spec_and_refuses_changed_measurement_scope(self):
        original = Path(__file__).resolve().parents[1] / bench.CASE_NAME
        spec = bench.load_case(original, 0)
        self.assertEqual(spec["seed"], 20260826)
        self.assertEqual(spec["scale"], .08838834764831843)
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".rgdr-latency-test-") as temp:
            Path(temp).resolve().relative_to(ROOT)
            path = Path(temp) / "cases.json"
            for change in ({"repeat_calls": 2}, {"state_dtype": "fp32"}, {"state_layout": "head_padded"},
                           {"input_layout": "noncontiguous"}, {"accepted_tokens": [1]}):
                modified = spec | change
                path.write_text(json.dumps([{"id": 0, "inputs": [{"name": "case_spec", "range_values": json.dumps(modified)}]}]))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    bench.load_case(path, 0)

    def test_kernel_sets_require_matching_names_content_and_nonempty_evidence(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".rgdr-latency-test-") as temp:
            Path(temp).resolve().relative_to(ROOT)
            runtime, built = Path(temp) / "runtime", Path(temp) / "built"
            name = "ascend950/recurrent_gated_delta_rule/test.o"
            for root in (runtime, built):
                path = root / name
                path.parent.mkdir(parents=True)
                path.write_bytes(b"same fixture")
            self.assertEqual(list(bench.compare_objects(runtime, built)), [name])
            (built / name).write_bytes(b"different fixture")
            with self.assertRaises(RuntimeError):
                bench.compare_objects(runtime, built)
            (built / name).unlink()
            with self.assertRaises(RuntimeError):
                bench.compare_objects(runtime, built)
            with self.assertRaises(RuntimeError):
                bench.target_objects(Path(temp) / "missing")

    def test_opapi_skip_or_different_runtime_cannot_count_as_proof(self):
        path = Path("installed-libcust_opapi.so").resolve()
        ok = "[OK] loaded libcust_opapi.so matches the freshly built one."
        bench.validate_opapi_check(f"loaded libcust_opapi.so: {path}\n{ok}", path)
        for text in (ok, f"loaded libcust_opapi.so: {path}\n[WARN] comparison skipped",
                     f"loaded libcust_opapi.so: other.so\n{ok}"):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                bench.validate_opapi_check(text, path)

    def test_profile_task_counts_and_duplicate_files_are_not_pooled(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".rgdr-latency-test-") as temp:
            Path(temp).resolve().relative_to(ROOT)
            root = Path(temp)
            for number in (1, 2):
                with (root / f"op_summary_{number}.csv").open("w", newline="", encoding="utf-8-sig") as file:
                    writer = csv.writer(file)
                    writer.writerow(["OP Type", "Task Duration(us)"])
                    writer.writerows([("RecurrentGatedDeltaRule", 4.2), ("RecurrentGatedDeltaRule", 4.4), ("Cast", 1.5)])
            evidence = bench.summarize_profile(root, 2)
            self.assertEqual(len(evidence), 2)
            self.assertEqual(evidence[0]["tasks"]["RecurrentGatedDeltaRule"]["count"], 2)
            self.assertAlmostEqual(evidence[0]["tasks"]["RecurrentGatedDeltaRule"]["mean_us"], 4.3)
            with self.assertRaises(RuntimeError):
                bench.summarize_profile(root, 4)
            with self.assertRaises(RuntimeError):
                bench.summarize_profile(root / "missing", 2)

    def test_statistics_reject_invalid_samples_and_preserve_zero_control(self):
        for values in ([], [float("nan")], [float("inf")], [-1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                bench.summarize(values)
        self.assertEqual(bench.summarize([0])["mean_us"], 0)
        self.assertEqual(bench.summarize([1, 2, 3])["median_us"], 2)

    def test_kernel_details_mapping_and_op_summary_are_not_pooled(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".rgdr-latency-test-") as temp:
            root = Path(temp).resolve()
            root.relative_to(ROOT)
            for name, headers in (("op_summary.csv", ["OP Type", "Task Duration(us)", "Task Wait Time(us)"]),
                                  ("kernel_details.csv", ["Type", "Duration(us)", "Wait Time(us)"])):
                with (root / name).open("w", newline="", encoding="utf-8-sig") as file:
                    writer = csv.writer(file)
                    writer.writerow(headers)
                    writer.writerows([("RecurrentGatedDeltaRule", 4.2, 1000), ("RecurrentGatedDeltaRule", 4.4, 2000)])
            evidence = bench.summarize_profile(root, 2)
            self.assertEqual(len(evidence), 2)
            self.assertEqual({e["duration_column"] for e in evidence}, {"Duration(us)", "Task Duration(us)"})
            for entry in evidence:
                self.assertEqual(entry["tasks"]["RecurrentGatedDeltaRule"]["count"], 2)
                self.assertAlmostEqual(entry["tasks"]["RecurrentGatedDeltaRule"]["mean_us"], 4.3)
            # A partial raw export cannot be counted as a complete run or pooled
            # with another view, but does not hide a complete kernel_details view.
            (root / "op_summary.csv").write_text("OP Type,Task Duration(us)\nRecurrentGatedDeltaRule,4.2\n")
            inventory = bench.profile_inventory(root)
            self.assertEqual(len(bench.summarize_profile(root, 2, inventory)), 1)
            self.assertEqual({e["status"] for e in inventory["csv_files"]}, {"accepted", "unexpected_rgdr_count"})

    def test_reported_na_type_uses_exact_name_and_original_duration_in_both_views(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".rgdr-latency-test-") as temp:
            root = Path(temp).resolve()
            root.relative_to(ROOT)
            # The three concrete durations supplied by the user, not a claim
            # about the mean of all twenty device samples.
            for name, headers in (("op_summary.csv", ["OP Type", "Op Name", "Task Duration(us)"]),
                                  ("kernel_details.csv", ["Type", "Name", "Duration(us)"])):
                with (root / name).open("w", newline="", encoding="utf-8-sig") as file:
                    writer = csv.writer(file)
                    writer.writerow(headers)
                    writer.writerows(("N/A", REPORTED_TASK_NAME, v) for v in (3.606, 3.605, 3.644))
            inventory = bench.profile_inventory(root)
            evidence = bench.summarize_profile(root, 3, inventory)
            self.assertEqual(len(evidence), 2)
            for entry in evidence:
                values = entry["tasks"]["RecurrentGatedDeltaRule"]
                self.assertEqual(values["count"], 3)
                self.assertAlmostEqual(values["mean_us"], 3.618333333333333)
                self.assertEqual(entry["rgdr_match_sources"], {"exact_op_name": 3})
            for entry in inventory["csv_files"]:
                self.assertEqual(entry["row_count"], 3)
                self.assertEqual(entry["type_counts"], {"N/A": 3})
                self.assertEqual(entry["rgdr_count"], 3)

    def test_name_fallback_rejects_substrings_placeholders_and_conflicting_types(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".rgdr-latency-test-") as temp:
            root = Path(temp).resolve()
            root.relative_to(ROOT)
            with (root / "kernel_details.csv").open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file)
                writer.writerow(["Type", "Name", "Duration(us)"])
                writer.writerows([
                    ("N/A", REPORTED_TASK_NAME, 3.606),
                    ("N/A", REPORTED_TASK_NAME + "Profile", 999),
                    ("N/A", "prefix_" + REPORTED_TASK_NAME, 999),
                    ("N/A", "PLACE_HOLDER_SQE", 999),
                    ("Cast", REPORTED_TASK_NAME, 999),
                ])
            evidence = bench.summarize_profile(root, 1)
            self.assertEqual(evidence[0]["tasks"]["RecurrentGatedDeltaRule"]["mean_us"], 3.606)
            self.assertEqual(evidence[0]["rgdr_match_sources"], {"exact_op_name": 1})
            with self.assertRaises(RuntimeError):
                bench.summarize_profile(root, 2)

    def test_unsupported_units_and_invalid_durations_retain_inventory(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".rgdr-latency-test-") as temp:
            root = Path(temp).resolve()
            root.relative_to(ROOT)
            table = root / "kernel_details.csv"
            table.write_text("Type,Duration(ms)\nRecurrentGatedDeltaRule,0.0042\n")
            with self.assertRaises(RuntimeError):
                bench.collect_profile_summary(root, 1, root)
            inventory = json.loads((root / "profile_inventory.json").read_text(encoding="utf-8"))
            self.assertEqual(inventory["csv_files"][0]["status"], "unsupported_columns_or_units")
            for value in ("nan", "inf", "-1", ""):
                table.write_text(f"Type,Duration(us)\nRecurrentGatedDeltaRule,{value}\n")
                with self.subTest(value=value), self.assertRaises(RuntimeError):
                    bench.collect_profile_summary(root, 1, root)
                inventory = json.loads((root / "profile_inventory.json").read_text(encoding="utf-8"))
                self.assertEqual(inventory["csv_files"][0]["status"], "invalid_duration")

    def test_offline_cli_preserves_previous_run_and_requires_no_online_environment(self):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix=".rgdr-latency-test-") as temp:
            root = Path(temp).resolve()
            root.relative_to(ROOT)
            previous = root / "result.json"
            previous.write_text('{"status":"failed","statistics":{"host_submit_us":{"mean_us":12.4}}}')
            original_bytes = previous.read_bytes()
            profile = root / "profile"
            profile.mkdir()
            table = profile / "kernel_details.csv"
            table.write_text(f"Type,Name,Duration(us)\nN/A,{REPORTED_TASK_NAME},4.2\nN/A,{REPORTED_TASK_NAME},4.4\n")
            parsed = root / "parsed"
            # No --run-package / --repo-root, and this also runs on Windows
            # without torch/NPU or a SIGALRM implementation.
            result = subprocess.run([sys.executable, "-B", str(Path(bench.__file__).resolve()),
                                     "--parse-profile", str(profile), "--profile-samples", "2",
                                     "--output", str(parsed)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("profile:", result.stdout)
            report = json.loads((parsed / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["mode"], "offline_profile_parse")
            self.assertEqual(previous.read_bytes(), original_bytes)
            self.assertTrue((parsed / "profile_inventory.json").is_file())


if __name__ == "__main__":
    unittest.main()

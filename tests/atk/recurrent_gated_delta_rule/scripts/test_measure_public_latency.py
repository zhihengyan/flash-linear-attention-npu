"""Offline control/evidence tests; these do not simulate NPU execution."""
from contextlib import nullcontext
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import measure_public_latency as bench

ROOT = Path(__file__).resolve().parents[4]


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
            def __enter__(self):
                self_outer.assertEqual(len(clones), 3)
                active[0] = True
                return self

            def __exit__(self, *args):
                active[0] = False

            def step(self):
                self_outer.assertTrue(active[0])

        self_outer = self
        def create_profile(**kwargs):
            self.assertFalse(kwargs["experimental_config"]["data_simplification"])
            return Capture()
        profiler = SimpleNamespace(
            ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
            ProfilerLevel=SimpleNamespace(Level0=0), _ExperimentalConfig=lambda **kw: kw,
            tensorboard_trace_handler=lambda path: path, profile=create_profile)
        torch = SimpleNamespace(profiler=SimpleNamespace(record_function=lambda name: nullcontext()))
        def call(state):
            self.assertTrue(active[0])
            self.assertIn(state, clones)
            submitted.append(state)
            return state
        with patch.object(bench, "npu_deadline", nullcontext):
            outputs, states = bench.profile_calls(torch, SimpleNamespace(profiler=profiler), call, State(),
                                                  lambda: None, 3, Path("unused-profile-path"))
        self.assertEqual(states, clones)
        self.assertEqual(outputs, submitted)
        self.assertEqual(len({id(s) for s in submitted}), 3)


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


if __name__ == "__main__":
    unittest.main()

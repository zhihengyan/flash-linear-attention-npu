"""Offline checks for msprof op evidence; no torch/NPU execution."""
import csv
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import msprof_op_rgdr as bench


ROOT = Path(__file__).resolve().parents[4]
KERNEL = "RecurrentGatedDeltaRule_85669048de2f4ae7785fd8d81b12f70b_266"
REPORTED_CSV = (
    "Op Name,Op Type,Task Duration(us),Block Dim,Mix Block Dim,Device Id,Pid,Current Freq,Rated Freq,\n"
    "RecurrentGatedDeltaRule_85669048de2f4ae7785fd8d81b12f70b_266,vector,5.125000,1,NA,1,8105,1650,1650,\n"
)


def table(path, rows, duration="Task Duration(us)"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Op Name", "Op Type", duration, "Device ID"])
        writer.writeheader()
        writer.writerows(rows)


def row(value, name=KERNEL, kind="N/A", device="1"):
    return {"Op Name": name, "Op Type": kind, "Task Duration(us)": value, "Device ID": device}


class MsprofEvidenceTest(unittest.TestCase):
    def test_application_clones_all_states_before_target_calls_with_plural_vendors(self):
        events = []

        class State:
            def clone(self):
                result = object()
                events.append(("clone", result))
                return result

        inputs = {key: object() for key in ("query", "key", "value", "beta", "scale",
                  "actual_seq_lengths", "ssm_state_indices", "num_accepted_tokens", "g", "gk")}
        inputs["state"] = State()
        stream = SimpleNamespace(npu_stream=17, synchronize=lambda: events.append(("sync",)))

        def build(spec, device):
            self.assertEqual((spec["HK"], spec["HV"]), (16, 48))
            events.append(("build",))
            return inputs

        def call(q, k, v, state, **kwargs):
            self.assertIs(q, inputs["query"])
            self.assertIs(k, inputs["key"])
            self.assertIs(v, inputs["value"])
            self.assertEqual(kwargs, {key: inputs[key] for key in (
                "beta", "scale", "actual_seq_lengths", "ssm_state_indices", "num_accepted_tokens", "g", "gk")})
            events.append(("call", state))
            return object()

        with tempfile.TemporaryDirectory() as temp:
            vendor = Path(temp) / "vendor"
            vendor.mkdir()
            library = vendor / "libcust_opapi.so"
            library.write_bytes(b"offline fixture, never loaded")
            fake_fla = SimpleNamespace(__file__=str(vendor / "__init__.py"),
                _prepare_embedded_opp=lambda: [vendor], _resolve_packaged_opapi=lambda path: library)
            op = SimpleNamespace(recurrent_gated_delta_rule=call)
            fake_torch = SimpleNamespace(__version__="fixture", device=lambda name: name,
                inference_mode=nullcontext, npu=SimpleNamespace(set_device=lambda device: None,
                get_device_name=lambda device: "Ascend950", current_stream=lambda: stream))
            modules = {"fla_npu": fake_fla, "torch": fake_torch,
                "torch_npu": SimpleNamespace(__version__="fixture"),
                "fla_npu.ops": SimpleNamespace(ascendc=op),
                "fla_npu.ops.ascendc": SimpleNamespace(_stable=SimpleNamespace(_loaded_path=None)),
                "executor_recurrent_gated_delta_rule": SimpleNamespace(build_inputs=build)}
            args = SimpleNamespace(repo_root=ROOT, case_id=1, device=0, calls=3,
                label="offline", metadata=Path(temp) / "application.json")
            with mock.patch.dict(sys.modules, modules), mock.patch.object(sys, "path", list(sys.path)):
                bench.run_application(args)
            report = json.loads(args.metadata.read_text(encoding="utf-8"))
            self.assertEqual(report["completed_calls"], 3)
            self.assertEqual(len(report["opapi_search_order"]), 1)
            self.assertEqual([item[0] for item in events],
                             ["build", "clone", "clone", "clone", "sync", "call", "sync", "call", "sync", "call", "sync"])
            self.assertEqual([item[1] for item in events if item[0] == "clone"],
                             [item[1] for item in events if item[0] == "call"])

    def test_multihead_perf_case_and_default_dry_run(self):
        path = ROOT / bench.OP_REL / bench.CASE_FILE
        original = path.read_bytes()
        spec = bench.load_case(ROOT, 1)
        self.assertEqual((spec["B"], spec["T"], spec["HK"], spec["HV"]), (1, 1, 16, 48))
        case = next(case for case in json.loads(original) if case["id"] == 1)
        attrs = {item["name"]: item["range_values"] for item in case["inputs"]}
        self.assertEqual((attrs["HK"], attrs["HV"]), (spec["HK"], spec["HV"]))
        self.assertEqual(spec["state_dtype"], "bf16")
        run = subprocess.run([sys.executable, "-I", str(Path(bench.__file__).resolve()), "--dry-run"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads(run.stdout)
        self.assertEqual(report["case_id"], 1)
        self.assertEqual(report["case_spec"], spec)
        self.assertEqual(report["calls"], 20)
        self.assertEqual(path.read_bytes(), original)

    def test_case_loader_rejects_shape_and_metadata_drift(self):
        cases = json.loads((ROOT / bench.OP_REL / bench.CASE_FILE).read_text(encoding="utf-8"))
        for key, value in (("HV", 32), ("seq_lengths", [0]), ("state_indices", [1]),
                           ("accepted_tokens", [1]), ("state_dtype", "fp32")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                case = json.loads(json.dumps(cases[1]))
                entry = next(item for item in case["inputs"] if item["name"] == "case_spec")
                spec = json.loads(entry["range_values"])
                spec[key] = value
                entry["range_values"] = json.dumps(spec)
                path = Path(temp) / bench.OP_REL / bench.CASE_FILE
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps([case]), encoding="utf-8")
                with self.assertRaises(ValueError):
                    bench.load_case(Path(temp), 1)

    def test_original_perf_case_is_reused_without_changes(self):
        spec = bench.load_case(ROOT, 0)
        self.assertEqual(spec["seed"], 20260826)
        self.assertEqual(spec["scale"], 0.08838834764831843)
        self.assertEqual(spec["seq_lengths"], [1])
        self.assertEqual(spec["state_indices"], [0])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            bench.load_case(ROOT, 999)

    def test_nested_per_call_tables_and_non_target_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            table(root / "OPPROF_run" / "RGDR" / "0" / "OpBasicInfo.csv",
                  [row(3.606), row(8, name="Neg", kind="Neg")])
            table(root / "OPPROF_run" / "RGDR" / "1" / "OpBasicInfo_20261103.csv", [row(3.644)])
            output = root / "summary.json"
            report = bench.summarize_msprof(root, 2, output)
            self.assertEqual(report["statistics"]["count"], 2)
            self.assertAlmostEqual(report["statistics"]["mean_us"], 3.625)
            self.assertAlmostEqual(report["statistics"]["p90_us"], 3.6402)
            self.assertEqual(len(report["samples"]), 2)
            self.assertEqual(json.loads(output.read_text())["status"], "complete")

    def test_unknown_types_need_precise_names_and_explicit_other_types_are_rejected(self):
        self.assertTrue(bench.is_rgdr(row(1)))
        self.assertTrue(bench.is_rgdr(row(1, name=bench.TASK_NAME, kind="")))
        self.assertTrue(bench.is_rgdr(row(1, name="any_launcher", kind="RecurrentGatedDeltaRule")))
        for name in (KERNEL + "Profile", "prefix_" + KERNEL, "RecurrentGatedDeltaRuleProfile", "PLACE_HOLDER_SQE"):
            self.assertFalse(bench.is_rgdr(row(1, name=name)))
            self.assertFalse(bench.is_rgdr(row(1, name=name, kind="vector")))
        self.assertTrue(bench.is_rgdr(row(1, kind="vector")))
        self.assertFalse(bench.is_rgdr(row(1, name="Neg", kind="vector")))
        self.assertFalse(bench.is_rgdr(row(1, kind="Cast")))

    def test_reported_vector_csv_is_recognized_without_inventing_twenty_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "OPPROF_run" / KERNEL / "0" / "OpBasicInfo_20261103140502018.csv"
            path.parent.mkdir(parents=True)
            path.write_text(REPORTED_CSV, encoding="utf-8")
            original = path.read_bytes()
            output = root / "summary.json"
            report = bench.summarize_msprof(root, 1, output)
            self.assertEqual(report["statistics"]["mean_us"], 5.125)
            self.assertEqual(report["statistics"]["count"], 1)
            self.assertEqual(report["devices"], ["1"])
            self.assertEqual(report["samples"][0]["row"]["Task Duration(us)"], "5.125000")
            self.assertEqual(report["samples"][0]["row"][""], "")
            self.assertEqual(report["inventory"][0]["rgdr_match_sources"], {"vector_kernel_name": 1})
            with self.assertRaisesRegex(ValueError, "expected 20 RGDR rows, got 1"):
                bench.summarize_msprof(root, 20, output)
            self.assertEqual(path.read_bytes(), original)

    def test_reported_device_id_spelling_and_mixed_aliases_reject_multiple_devices(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, second = root / "0" / "OpBasicInfo.csv", root / "1" / "OpBasicInfo.csv"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_text(REPORTED_CSV, encoding="utf-8")
            second.write_text(REPORTED_CSV.replace(",NA,1,8105,", ",NA,0,8105,"), encoding="utf-8")
            output = root / "summary.json"
            with self.assertRaisesRegex(ValueError, "multiple devices"):
                bench.summarize_msprof(root, 2, output)
            # Mixed header spellings must not hide another device either.
            table(second, [row(5.125, kind="vector", device="0")])
            with self.assertRaisesRegex(ValueError, "multiple devices"):
                bench.summarize_msprof(root, 2, output)

    def test_partial_duplicate_and_multi_run_evidence_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "OPPROF_first" / "OpBasicInfo.csv"
            table(target, [row(3.6)])
            output = root / "summary.json"
            with self.assertRaisesRegex(ValueError, "expected 2 RGDR rows, got 1"):
                bench.summarize_msprof(root, 2, output)
            table(root / "OPPROF_first" / "duplicate" / "OpBasicInfo.csv", [row(3.6)])
            with self.assertRaisesRegex(ValueError, "expected 1 RGDR rows, got 2"):
                bench.summarize_msprof(root, 1, output)
            table(root / "OPPROF_second" / "OpBasicInfo.csv", [row(3.6)])
            with self.assertRaisesRegex(ValueError, "multiple OPPROF"):
                bench.summarize_msprof(root, 3, output)
            self.assertEqual(json.loads(output.read_text())["status"], "failed")

    def test_bad_units_nonfinite_values_and_multiple_devices_fail_with_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, output = root / "OpBasicInfo.csv", root / "summary.json"
            table(path, [{"Op Name": KERNEL, "Op Type": "N/A", "Duration(ms)": .0036, "Device ID": "1"}],
                  duration="Duration(ms)")
            with self.assertRaisesRegex(ValueError, "schema/units"):
                bench.summarize_msprof(root, 1, output)
            for value in ("nan", "inf", -1, 0, "N/A"):
                with self.subTest(value=value):
                    table(path, [row(value)])
                    with self.assertRaises(ValueError):
                        bench.summarize_msprof(root, 1, output)
                    report = json.loads(output.read_text())
                    self.assertEqual(report["status"], "failed")
                    self.assertEqual(report["inventory"][0]["file"], str(path.resolve()))
            table(path, [row(3.6), row(3.7, device="0")])
            with self.assertRaisesRegex(ValueError, "multiple devices"):
                bench.summarize_msprof(root, 2, output)

    def test_zero_matches_keep_original_rows_for_diagnosis(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, output = root / "OpBasicInfo.csv", root / "summary.json"
            original = row(3.606, name="unrecognized_kernel", kind="unrecognized_type")
            table(path, [original])
            with self.assertRaisesRegex(ValueError, "unmatched_row_examples"):
                bench.summarize_msprof(root, 20, output)
            report = json.loads(output.read_text())
            entry = report["inventory"][0]
            self.assertEqual(report["status"], "failed")
            self.assertEqual(entry["row_count"], 1)
            self.assertEqual(entry["type_counts"], {"unrecognized_type": 1})
            self.assertEqual(entry["unmatched_row_examples"][0]["row"]["Task Duration(us)"], "3.606")
            self.assertEqual(entry["unmatched_row_examples"][0]["row"]["Op Name"], original["Op Name"])

    def test_offline_cli_does_not_require_torch_repo_or_npu_and_preserves_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "OPPROF_run" / "OpBasicInfo.csv"
            path.parent.mkdir(parents=True)
            path.write_text(REPORTED_CSV, encoding="utf-8")
            original = path.read_bytes()
            run = subprocess.run([sys.executable, "-I", str(Path(bench.__file__).resolve()),
                                  "--summarize", str(root), "--expected-count", "1"],
                                 cwd=root, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("count=1", run.stdout)
            self.assertIn("mean_us=5.12500", run.stdout)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()

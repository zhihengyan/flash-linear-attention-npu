"""Offline checks for msprof op evidence; no torch/NPU execution."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

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

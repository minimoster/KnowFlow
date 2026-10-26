import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import rag_eval as ev
import rag_experiment as experiment


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "inputs"
        ev.fixture(self.inputs)
        self.dataset = self.inputs / "labels.jsonl"
        self.corpus = self.inputs / "corpus.jsonl"
        self.runs = [self.inputs / f"{mode}.json" for mode in ev.MODES]
        self.output = self.root / "experiment"

    def pack(self, **kwargs):
        return experiment.pack(self.dataset, self.corpus, self.runs, self.output, fixture=True, **kwargs)

    def mutate_run(self, mode, update):
        path = self.inputs / f"{mode}.json"
        data = experiment.load_json(path)
        update(data)
        ev.write_json(path, data)

    def refresh_dataset_hash(self):
        for mode in ev.MODES:
            self.mutate_run(mode, lambda run: run.update(dataset_sha256=ev.digest(self.dataset)))

    def rehash_archived_file(self, name):
        path = self.output / "manifest.json"
        manifest = experiment.load_json(path)
        manifest["files"][name] = {"sha256": ev.digest(self.output / name),
                                    "bytes": (self.output / name).stat().st_size}
        ev.write_json(path, manifest)

    def test_pack_roundtrip_keeps_inputs_and_fixture_classification(self):
        original = {path: path.read_bytes() for path in [self.dataset, self.corpus, *self.runs]}
        manifest = self.pack()
        self.assertEqual(manifest["classification"], experiment.FIXTURE)
        self.assertEqual(experiment.verify(self.output)["integrity"], "consistent")
        for path, data in original.items():
            self.assertEqual(path.read_bytes(), data)
            self.assertEqual((self.output / path.name).read_bytes(), data)
        self.assertIn(experiment.FIXTURE, (self.output / "REPORT.md").read_text())

    def test_fixture_never_implicitly_becomes_real(self):
        with self.assertRaisesRegex(ValueError, "requires label_status=reviewed"):
            experiment.pack(self.dataset, self.corpus, self.runs, self.output)
        self.assertFalse(self.output.exists())

    def test_pending_review_does_not_create_an_archive(self):
        rows = ev.read_jsonl(self.dataset)
        rows[0]["label_status"] = "pending_review"
        ev.write_jsonl(self.dataset, rows)
        self.refresh_dataset_hash()
        with self.assertRaises(ValueError):
            self.pack()
        self.assertFalse(self.output.exists())

    def test_output_is_never_overwritten(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("existing experiment")
        with self.assertRaises(FileExistsError):
            self.pack()
        self.assertEqual(marker.read_text(), "existing experiment")
        self.assertEqual(list(self.output.iterdir()), [marker])

    def test_changed_raw_export_is_detected(self):
        self.pack()
        with (self.output / "vector.json").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "fingerprint differs"):
            experiment.verify(self.output)

    def test_rehashed_score_edits_still_fail_recomputation(self):
        self.pack()
        path = self.output / "report.json"
        report = experiment.load_json(path)
        report["modes"]["vector"]["recall_at_5"] = 0.125
        ev.write_json(path, report)
        self.rehash_archived_file("report.json")
        with self.assertRaisesRegex(ValueError, "recomputed raw exports"):
            experiment.verify(self.output)

    def test_rehashed_markdown_edits_still_fail_recomputation(self):
        self.pack()
        path = self.output / "REPORT.md"
        path.write_text(path.read_text() + "Unmeasured improvement claim\n")
        self.rehash_archived_file("REPORT.md")
        with self.assertRaisesRegex(ValueError, "recomputed raw exports"):
            experiment.verify(self.output)

    def test_manifest_cannot_relabel_fixture_as_real(self):
        self.pack()
        path = self.output / "manifest.json"
        manifest = experiment.load_json(path)
        manifest["classification"] = experiment.REAL
        ev.write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "requires label_status=reviewed"):
            experiment.verify(self.output)

    def test_missing_or_extra_bundle_files_refused(self):
        self.pack()
        extra = self.output / "extra.json"
        extra.write_text("{}")
        with self.assertRaisesRegex(ValueError, "exactly"):
            experiment.verify(self.output)
        extra.unlink()
        (self.output / "hybrid.json").unlink()
        with self.assertRaisesRegex(ValueError, "exactly"):
            experiment.verify(self.output)

    def test_reporter_version_drift_is_explicit(self):
        self.pack()
        path = self.output / "manifest.json"
        manifest = experiment.load_json(path)
        manifest["tools_sha256"]["rag_eval.py"] = "0" * 64
        ev.write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "scoring scripts changed"):
            experiment.verify(self.output)

    def test_missing_cost_stays_unknown_but_measured_zero_is_known(self):
        self.mutate_run("hybrid", lambda run: run["queries"][0].update(cost_usd=0.0))
        self.pack()
        modes = experiment.load_json(self.output / "comparison.json")["modes"]
        self.assertIsNone(modes["vector"]["cost"]["total_usd"])
        self.assertIsNone(modes["vector"]["cost"]["known_subtotal_usd"])
        self.assertEqual(modes["hybrid"]["cost"]["coverage"], 1)
        self.assertEqual(modes["hybrid"]["cost"]["total_usd"], 0.0)

    def test_partial_cost_is_not_reported_as_total(self):
        rows = ev.read_jsonl(self.dataset)
        other = copy.deepcopy(rows[0])
        other["query_id"] = "fixture-2"
        ev.write_jsonl(self.dataset, [*rows, other])
        for mode in ev.MODES:
            def update(run):
                second = copy.deepcopy(run["queries"][0])
                second["query_id"] = "fixture-2"
                run["queries"] += [second]
                run["queries"][0]["cost_usd"] = 0.25
                run["dataset_sha256"] = ev.digest(self.dataset)
            self.mutate_run(mode, update)
        self.pack()
        cost = experiment.load_json(self.output / "comparison.json")["modes"]["vector"]["cost"]
        self.assertEqual(cost["known_subtotal_usd"], 0.25)
        self.assertEqual(cost["known_queries"], 1)
        self.assertEqual(cost["unknown_queries"], 1)
        self.assertIsNone(cost["total_usd"])

    def test_invalid_cost_values_rejected(self):
        for bad in (-1, True, "0.01"):
            with self.subTest(cost=bad):
                self.mutate_run("vector", lambda run: run["queries"][0].update(cost_usd=bad))
                with self.assertRaisesRegex(ValueError, "cost_usd"):
                    self.pack()
                self.assertFalse(self.output.exists())

    def test_cost_overflow_returns_clean_cli_refusal(self):
        rows = ev.read_jsonl(self.dataset)
        other = copy.deepcopy(rows[0])
        other["query_id"] = "fixture-2"
        ev.write_jsonl(self.dataset, [*rows, other])
        for mode in ev.MODES:
            def update(run):
                run["queries"][0]["cost_usd"] = 1e308
                second = copy.deepcopy(run["queries"][0])
                second["query_id"] = "fixture-2"
                run["queries"].append(second)
                run["dataset_sha256"] = ev.digest(self.dataset)
            self.mutate_run(mode, update)
        result = subprocess.run([sys.executable, str(Path(experiment.__file__).resolve()), "pack",
                                 "--dataset", str(self.dataset), "--corpus", str(self.corpus),
                                 "--runs", *map(str, self.runs), "--output-dir", str(self.output), "--fixture"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("cost_usd total exceeds", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(self.output.exists())

    def test_switching_baseline_reverses_paired_metric_delta(self):
        self.mutate_run("hybrid", lambda run: run["queries"][0].update(ranked_chunk_ids=["a", "b", "c"]))
        self.pack(baseline="hybrid")
        summary = experiment.load_json(self.output / "comparison.json")
        comparison = summary["comparisons"]["vector"]
        self.assertEqual(comparison["mrr_at_5_delta"], -0.5)
        self.assertEqual(comparison["per_query"][0]["ranking_change"], "regressed")
        self.assertNotIn("hybrid", summary["comparisons"])

    def test_failure_recovery_is_kept_even_when_both_scores_are_zero(self):
        self.mutate_run("vector", lambda run: run["queries"][0].update(ranked_chunk_ids=[], error="TimeoutError"))
        self.mutate_run("hybrid", lambda run: run["queries"][0].update(ranked_chunk_ids=[]))
        self.pack()
        summary = experiment.load_json(self.output / "comparison.json")
        pair = summary["comparisons"]["hybrid"]["per_query"][0]
        self.assertEqual(pair["ranking_change"], "unchanged")
        self.assertEqual(pair["execution_transition"], "recovered_error")
        self.assertEqual(pair["baseline_outcome"], "execution_error")
        self.assertEqual(pair["candidate_outcome"], "empty_retrieval")
        self.assertEqual(summary["modes"]["vector"]["questions"], 1)
        self.assertEqual(summary["modes"]["vector"]["recall_at_5"], 0)

    def test_recall_mrr_tradeoff_is_not_labeled_unqualified_improvement(self):
        rows = ev.read_jsonl(self.dataset)
        text = "Synthetic evidence c."
        rows[0]["evidence"].append({"chunk_id": "c", "start": 0, "end": len(text), "quote": text,
                                    "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                    "relevance": 3, "rationale": "Synthetic second relevant chunk"})
        ev.write_jsonl(self.dataset, rows)
        self.refresh_dataset_hash()
        self.mutate_run("vector", lambda run: run["queries"][0].update(ranked_chunk_ids=["a", "b"]))
        self.pack()
        pair = experiment.load_json(self.output / "comparison.json")["comparisons"]["hybrid"]["per_query"][0]
        self.assertEqual(pair["recall_at_5_delta"], 0.5)
        self.assertEqual(pair["mrr_at_5_delta"], -0.5)
        self.assertEqual(pair["ranking_change"], "tradeoff")

    def test_mixed_development_test_splits_rejected(self):
        rows = ev.read_jsonl(self.dataset)
        other = copy.deepcopy(rows[0])
        other.update(query_id="fixture-test", split="test")
        ev.write_jsonl(self.dataset, [*rows, other])
        for mode in ev.MODES:
            def update(run):
                extra = copy.deepcopy(run["queries"][0])
                extra["query_id"] = "fixture-test"
                run["queries"].append(extra)
                run["dataset_sha256"] = ev.digest(self.dataset)
            self.mutate_run(mode, update)
        with self.assertRaisesRegex(ValueError, "splits separately"):
            self.pack()
        self.assertFalse(self.output.exists())

    def test_missing_query_and_shared_config_drift_still_rejected(self):
        original = self.runs[0].read_bytes()
        for update in (lambda run: run.update(queries=[]), lambda run: run["config"].update(candidate_k=99)):
            self.runs[0].write_bytes(original)
            self.mutate_run("vector", update)
            with self.assertRaises(ValueError):
                self.pack()
            self.assertFalse(self.output.exists())

    def test_placeholder_revision_not_accepted_for_real_provenance(self):
        report = experiment.load_json(self.inputs / "report.json")
        rows = ev.read_jsonl(self.dataset)
        with self.assertRaisesRegex(ValueError, "tested revision SHA"):
            experiment.check_provenance(report, rows, fixture=False)

    def test_cli_pack_and_verify_use_existing_exports(self):
        script = str(Path(experiment.__file__).resolve())
        args = [sys.executable, script, "pack", "--dataset", str(self.dataset), "--corpus", str(self.corpus),
                "--runs", *map(str, self.runs), "--output-dir", str(self.output), "--fixture"]
        packed = subprocess.run(args, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(packed.stdout)["classification"], experiment.FIXTURE)
        verified = subprocess.run([sys.executable, script, "verify", "--bundle", str(self.output)],
                                  capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(verified.stdout)["integrity"], "consistent")
        refused = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(refused.returncode, 2)
        self.assertIn("already exists", refused.stderr)


if __name__ == "__main__":
    unittest.main()

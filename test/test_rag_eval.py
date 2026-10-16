import copy
import json
from pathlib import Path
import tempfile
import sys
import types
import unittest

import rag_eval as ev


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.report = ev.fixture(self.root)
        self.dataset = self.root / "labels.jsonl"
        self.corpus = self.root / "corpus.jsonl"
        self.paths = [self.root / f"{m}.json" for m in ev.MODES]

    def compare(self):
        return ev.compare(self.dataset,self.corpus,self.paths,self.root / "report.json",fixture=True)

    def mutate_run(self, update):
        value = json.loads(self.paths[0].read_text(encoding="utf-8"))
        update(value)
        ev.write_json(self.paths[0],value)

    def test_metric_arithmetic_and_cutoff(self):
        result = ev.metrics(["b","a","c","d","e","f"],{"a","f"})
        self.assertEqual(result["recall_at_5"],.5)
        self.assertEqual(result["mrr_at_5"],.5)
        self.assertEqual(ev.metrics(["b"],{"a"})["mrr_at_5"],0)

    def test_duplicate_results_rejected(self):
        with self.assertRaises(ValueError): ev.metrics(["a","a"],{"a"})

    def test_empty_gold_rejected(self):
        with self.assertRaises(ValueError): ev.metrics([],set())

    def test_fixture_label_and_identical_groups(self):
        self.assertEqual(self.report["kind"],"OFFLINE_FIXTURE_NOT_REAL_RETRIEVAL")
        for mode in ev.MODES:
            self.assertEqual(self.report["modes"][mode]["recall_at_5"],1)
            self.assertEqual(self.report["modes"][mode]["mrr_at_5"],.5)

    def test_fixture_cannot_be_used_as_real_gold(self):
        with self.assertRaises(ValueError): ev.validate(self.dataset,self.corpus)

    def test_legacy_labels_remain_pending(self):
        legacy = self.root / "legacy.jsonl"
        ev.write_jsonl(legacy,[{"query_id":f"q{i}","question":"q","ground_truth":["a"]} for i in range(40)])
        queue = self.root / "queue.jsonl"
        ev.prepare(legacy,queue,self.corpus)
        rows = ev.read_jsonl(queue)
        self.assertEqual(len(rows),40)
        self.assertTrue(all(x["label_status"] == "pending_review" and x["evidence"] == [] for x in rows))
        self.assertTrue(all(x["candidate_evidence"][0]["relevance"] is None for x in rows))
        with self.assertRaises(ValueError): ev.validate(queue,self.corpus)

    def test_tampered_evidence_refused(self):
        for key, value in [("quote","invented"),("content_sha256","wrong"),("end",999),("chunk_id","unknown")]:
            with self.subTest(key=key):
                rows = ev.read_jsonl(self.dataset)
                changed = copy.deepcopy(rows); changed[0]["evidence"][0][key] = value
                ev.write_jsonl(self.dataset,changed)
                with self.assertRaises(ValueError): ev.validate(self.dataset,self.corpus,fixture=True)
                ev.write_jsonl(self.dataset,rows)

    def test_corpus_drift_refused(self):
        self.corpus.write_text(self.corpus.read_text(encoding="utf-8") + "\n",encoding="utf-8")
        with self.assertRaises(ValueError): self.compare()

    def test_config_drift_refused(self):
        self.mutate_run(lambda x: x["config"].update(candidate_k=99))
        with self.assertRaises(ValueError): self.compare()

    def test_missing_query_refused(self):
        self.mutate_run(lambda x: x.update(queries=[]))
        with self.assertRaises(ValueError): self.compare()

    def test_unknown_id_refused(self):
        self.mutate_run(lambda x: x["queries"][0].update(ranked_chunk_ids=["missing"]))
        with self.assertRaises(ValueError): self.compare()

    def test_failed_query_counts_as_zero(self):
        self.mutate_run(lambda x: x["queries"][0].update(ranked_chunk_ids=[],error="TimeoutError"))
        result = self.compare()["modes"]["vector"]
        self.assertEqual(result["recall_at_5"],0)
        self.assertEqual(result["failed_queries"],1)
        self.assertEqual(result["questions"],1)

    def test_failed_query_with_ranking_refused(self):
        self.mutate_run(lambda x: x["queries"][0].update(error="TimeoutError"))
        with self.assertRaises(ValueError): self.compare()

    def test_invalid_latency_refused(self):
        self.mutate_run(lambda x: x["queries"][0].update(latency_ms=-1))
        with self.assertRaises(ValueError): self.compare()

    def test_fixture_cannot_be_used_as_real_run(self):
        self.mutate_run(lambda x: x.update(kind="retrieval"))
        with self.assertRaises(ValueError): self.compare()

    def reviewed_test_input(self):
        # Only inside this test's temp directory: exercise the real-input schema.
        rows = ev.read_jsonl(self.dataset)
        rows[0].update(label_status="reviewed",reviewed_by="synthetic unittest",reviewed_at="2026-10-16T00:00:00Z")
        ev.write_jsonl(self.dataset,rows)
        config = json.loads(self.paths[2].read_text(encoding="utf-8"))["config"]
        path = self.root / "config.json"; ev.write_json(path,config)
        return path

    def test_collect_runs_all_modes_and_preserves_errors(self):
        config = self.reviewed_test_input()
        seen = []
        module = types.ModuleType("offline_adapter_test")
        def retrieve(question,mode,config):
            seen.append(mode)
            if mode == "hybrid": raise TimeoutError("must not serialize credentials or error text")
            return {"ranked_chunk_ids":["a"]}
        module.retrieve = retrieve
        sys.modules[module.__name__] = module
        self.addCleanup(sys.modules.pop,module.__name__)
        report = ev.collect(self.dataset,self.corpus,module.__name__+":retrieve",config,self.root / "collected")
        self.assertEqual(set(seen),set(ev.MODES))
        row = report["modes"]["hybrid"]["per_query"][0]
        self.assertEqual(row["error"],"TimeoutError")
        self.assertEqual(row["recall_at_5"],0)
        self.assertIsNone(row["cost_usd"])

    def test_collect_rejects_invalid_config_before_adapter_import(self):
        config = self.reviewed_test_input()
        ev.write_json(config,{})
        with self.assertRaises(ValueError):
            ev.collect(self.dataset,self.corpus,"nonexistent_module:retrieve",config,self.root / "collected")


if __name__ == "__main__":
    unittest.main()

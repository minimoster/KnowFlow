import json
from pathlib import Path
import tempfile
import unittest

import rag_eval as ev
import rag_review as review


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);ev.fixture(self.root)
        self.packet=self.root / "packet.json"
        review.pool(self.root/"corpus.jsonl",self.root/"labels.jsonl",[self.root/f"{m}.json" for m in ev.MODES],self.packet)
        self.a=self.root/"a.jsonl";self.b=self.root/"b.jsonl";self.output=self.root/"consensus.jsonl"

    def submissions(self):
        for name,path in (("synthetic reviewer A",self.a),("synthetic reviewer B",self.b)):
            review.template(self.packet,name,path)
            rows=ev.read_jsonl(path)
            for row in rows:
                row.update(complete=True,reviewed_at="2026-10-16T00:00:00Z")
                for item in row["judgments"]:
                    text=f"Synthetic evidence {item['chunk_id']}."
                    item.update(relevance=3 if item["chunk_id"]=="a" else 0,rationale="synthetic test judgment",start=0,end=len(text),quote=text)
            ev.write_jsonl(path,rows)

    def test_packet_blinds_arms_ranks_and_old_grades(self):
        packet=json.loads(self.packet.read_text(encoding="utf-8"))
        self.assertEqual(packet["kind"],"synthetic_review_packet")
        for candidate in packet["questions"][0]["candidates"]:
            self.assertEqual(set(candidate),{"chunk_id","content","content_sha256"})

    def test_template_is_not_reviewed(self):
        review.template(self.packet,"test",self.a)
        row=ev.read_jsonl(self.a)[0]
        self.assertFalse(row["complete"]);self.assertIsNone(row["reviewed_at"])
        self.assertTrue(all(x["relevance"] is None for x in row["judgments"]))

    def test_synthetic_consensus_stays_fixture(self):
        self.submissions();result=review.reconcile(self.packet,self.a,self.b,self.output)
        self.assertEqual(result["reviewed"],0);self.assertEqual(result["synthetic"],1)
        ev.validate(self.output,self.root/"corpus.jsonl",fixture=True)
        with self.assertRaises(ValueError):ev.validate(self.output,self.root/"corpus.jsonl")

    def test_same_reviewer_rejected(self):
        self.submissions();rows=ev.read_jsonl(self.b);rows[0]["reviewer"]="synthetic reviewer A";ev.write_jsonl(self.b,rows)
        with self.assertRaises(ValueError):review.reconcile(self.packet,self.a,self.b,self.output)

    def test_disagreement_needs_adjudication(self):
        self.submissions();rows=ev.read_jsonl(self.b)
        rows[0]["judgments"][0]["relevance"]=2
        ev.write_jsonl(self.b,rows)
        result=review.reconcile(self.packet,self.a,self.b,self.output)
        self.assertEqual(result["needs_adjudication"],1)
        with self.assertRaises(ValueError):ev.validate(self.output,self.root/"corpus.jsonl",fixture=True)

    def test_missing_judgments_rejected(self):
        self.submissions();rows=ev.read_jsonl(self.a);rows[0]["judgments"].pop();ev.write_jsonl(self.a,rows)
        with self.assertRaises(ValueError):review.reconcile(self.packet,self.a,self.b,self.output)

    def test_fabricated_support_quote_rejected(self):
        self.submissions();rows=ev.read_jsonl(self.a)
        for item in rows[0]["judgments"]:
            if item["relevance"]>0:item["quote"]="invented"
        ev.write_jsonl(self.a,rows)
        with self.assertRaises(ValueError):review.reconcile(self.packet,self.a,self.b,self.output)

    def test_failure_triage_preserves_evidence_and_remains_unreviewed(self):
        report=json.loads((self.root/"report.json").read_text(encoding="utf-8"))
        row=report["modes"]["vector"]["per_query"][0];row.update(recall_at_5=0,ranked_chunk_ids=["b"])
        ev.write_json(self.root/"report.json",report)
        result=review.failures(self.root/"report.json",self.root/"labels.jsonl",self.root/"corpus.jsonl",self.output)
        self.assertEqual(result["failure_cases"],1)
        row=ev.read_jsonl(self.output)[0]
        self.assertEqual(row["triage_status"],"unreviewed")
        self.assertEqual(row["missing_judged_evidence"],["a"])


if __name__=="__main__":unittest.main()

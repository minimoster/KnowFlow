#!/usr/bin/env python3
"""Blind evidence pooling, two-submission reconciliation and failure triage.

These tools never supply relevance judgments or certify human identity. Real
reviewer submissions are still required. Synthetic packets stay synthetic.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import rag_eval as ev


def pool(corpus, questions, run_paths, output):
    chunks = ev.unique(ev.read_jsonl(corpus),"chunk_id")
    rows = ev.read_jsonl(questions); ev.unique(rows,"query_id")
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in run_paths]
    if len(runs)!=3 or {r.get("mode") for r in runs}!=set(ev.MODES):
        raise ValueError("pool requires all three retrieval arms")
    kinds={r.get("kind") for r in runs}
    if kinds not in ({"retrieval"},{"fixture"}): raise ValueError("mixed/unknown run kinds")
    for run in runs:
        if run.get("dataset_sha256")!=ev.digest(questions) or run.get("corpus_sha256")!=ev.digest(corpus):
            raise ValueError("pool source fingerprints differ")
    rankings=[ev.unique(r["queries"],"query_id") for r in runs]
    if any(set(r)!=set(x["query_id"] for x in rows) for r in rankings):
        raise ValueError("pool is missing query results")
    packet={"kind":"synthetic_review_packet" if kinds=={"fixture"} else "review_packet",
            "corpus_sha256":ev.digest(corpus),"questions_sha256":ev.digest(questions),
            "source_run_sha256":sorted(ev.digest(p) for p in run_paths),"questions":[]}
    for row in rows:
        qid=row["query_id"]
        ids={cid for arm in rankings for cid in arm[qid]["ranked_chunk_ids"]}
        ids.update(x["chunk_id"] for x in row.get("candidate_evidence",[]))
        ids.update(x["chunk_id"] for x in row.get("evidence",[]))
        candidates=[]
        for cid in sorted(ids,key=lambda cid:hashlib.sha256((qid+"\0"+cid).encode()).hexdigest()):
            if cid not in chunks: raise ValueError("unknown pooled chunk ID")
            text=chunks[cid]["content"]
            candidates.append({"chunk_id":cid,"content":text,
                               "content_sha256":hashlib.sha256(text.encode()).hexdigest()})
        if not candidates: raise ValueError("empty evidence pool; add independent candidates before reviewing")
        packet["questions"].append({"query_id":qid,"question":row["question"],
                                    "split":row.get("split","development"),"candidates":candidates})
    # No source arm, score, rank, old grade or expected answer is revealed here.
    ev.write_json(output,packet); return packet


def template(packet_path, reviewer, output):
    packet=json.loads(Path(packet_path).read_text(encoding="utf-8"))
    rows=[]
    for row in packet["questions"]:
        rows.append({"query_id":row["query_id"],"packet_sha256":ev.digest(packet_path),
                     "reviewer":reviewer,"reviewed_at":None,"complete":False,
                     "judgments":[{"chunk_id":c["chunk_id"],"relevance":None,
                                   "start":None,"end":None,"quote":None,"rationale":None}
                                  for c in row["candidates"]]})
    ev.write_jsonl(output,rows)


def check_submission(row, question, packet_hash):
    if row.get("complete") is not True or row.get("packet_sha256")!=packet_hash:
        raise ValueError("incomplete review or packet fingerprint mismatch")
    reviewer=row.get("reviewer")
    if not isinstance(reviewer,str) or not reviewer.strip(): raise ValueError("reviewer missing")
    try: timestamp=datetime.fromisoformat(row["reviewed_at"].replace("Z","+00:00"))
    except (KeyError,TypeError,ValueError,AttributeError): raise ValueError("valid reviewed_at timestamp required")
    if timestamp.tzinfo is None: raise ValueError("review timestamp needs timezone")
    judgments=ev.unique(row.get("judgments",[]),"chunk_id")
    candidates=ev.unique(question["candidates"],"chunk_id")
    if judgments.keys()!=candidates.keys(): raise ValueError("every pooled chunk must be independently judged")
    evidence=[]
    for cid,candidate in candidates.items():
        item=dict(judgments[cid]); grade=item.get("relevance"); text=candidate["content"]
        if type(grade) is not int or not 0<=grade<=3: raise ValueError("review relevance must be integer 0–3")
        if not isinstance(item.get("rationale"),str) or not item["rationale"].strip(): raise ValueError("every judgment needs rationale")
        if grade>0:
            start,end=item.get("start"),item.get("end")
            if type(start) is not int or type(end) is not int or not 0<=start<end<=len(text) or text[start:end]!=item.get("quote"):
                raise ValueError("positive judgment must cite exact supporting span")
        else:
            # For negative judgments preserve the assessed text, not a fabricated support span.
            item.update(start=0,end=len(text),quote=text)
        item["content_sha256"]=candidate["content_sha256"];evidence.append(item)
    return reviewer.strip(),timestamp,evidence


def reconcile(packet_path, reviews_a, reviews_b, output):
    packet=json.loads(Path(packet_path).read_text(encoding="utf-8")); packet_hash=ev.digest(packet_path)
    if packet.get("kind") not in ("review_packet","synthetic_review_packet"): raise ValueError("unknown packet kind")
    a=ev.unique(ev.read_jsonl(reviews_a),"query_id");b=ev.unique(ev.read_jsonl(reviews_b),"query_id")
    if a.keys()!=b.keys() or a.keys()!=set(q["query_id"] for q in packet["questions"]): raise ValueError("review question mismatch")
    rows=[]
    for question in packet["questions"]:
        qid=question["query_id"]
        reviewer_a,time_a,evidence_a=check_submission(a[qid],question,packet_hash)
        reviewer_b,time_b,evidence_b=check_submission(b[qid],question,packet_hash)
        if reviewer_a.casefold()==reviewer_b.casefold(): raise ValueError("independent submissions need different reviewer identifiers")
        grades_b={e["chunk_id"]:e["relevance"] for e in evidence_b}
        disagreements=[e["chunk_id"] for e in evidence_a if grades_b[e["chunk_id"]]!=e["relevance"]]
        positive=any(e["relevance"]>0 for e in evidence_a)
        status="needs_adjudication" if disagreements or not positive else "reviewed"
        if status=="reviewed" and packet["kind"]=="synthetic_review_packet": status="fixture"
        rows.append({"query_id":qid,"question":question["question"],"split":question["split"],
                     "corpus_sha256":packet["corpus_sha256"],"label_status":status,
                     "reviewed_by":reviewer_a+" + "+reviewer_b,"reviewed_at":max(time_a,time_b).isoformat(),
                     "evidence":evidence_a,"disagreements":disagreements,
                     "review_a_sha256":ev.digest(reviews_a),"review_b_sha256":ev.digest(reviews_b),
                     "review_b_evidence":evidence_b,"review_packet_sha256":packet_hash,
                     "review_notes":"Identity strings and supplied judgments validated; tool does not certify human identity. Adjudicate disagreements; never choose the better-scoring label."})
    ev.write_jsonl(output,rows)
    return {"questions":len(rows),"reviewed":sum(r["label_status"]=="reviewed" for r in rows),
            "synthetic":sum(r["label_status"]=="fixture" for r in rows),
            "needs_adjudication":sum(r["label_status"]=="needs_adjudication" for r in rows)}


def failures(report_path, dataset, corpus, output):
    report=json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("dataset_sha256")!=ev.digest(dataset) or report.get("corpus_sha256")!=ev.digest(corpus):
        raise ValueError("failure report fingerprints differ")
    rows,chunks=ev.validate(dataset,corpus,fixture=report["kind"]=="OFFLINE_FIXTURE_NOT_REAL_RETRIEVAL")
    questions=ev.unique(rows,"query_id");cases=[]
    for mode,arm in report["modes"].items():
        for result in arm["per_query"]:
            if not result.get("error") and result["recall_at_5"]==1:continue
            row=questions[result["query_id"]]
            gold={e["chunk_id"] for e in row["evidence"] if e["relevance"]>0}
            cases.append({"kind":report["kind"],"query_id":row["query_id"],"question":row["question"],
                          "mode":mode,"config":arm["config"],"error":result.get("error"),
                          "recall_at_5":result["recall_at_5"],"missing_judged_evidence":sorted(gold-set(result["ranked_chunk_ids"])),
                          "retrieved_chunks":[chunks[cid] for cid in result["ranked_chunk_ids"]],
                          "evidence":row["evidence"],"triage_status":"unreviewed",
                          "hypothesis":None,"action":None,"retest_revision":None})
    ev.write_jsonl(output,cases);return {"failure_cases":len(cases)}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
    s=sub.add_parser("pool");s.add_argument("--corpus",required=True);s.add_argument("--questions",required=True);s.add_argument("--runs",nargs=3,required=True)
    s=sub.add_parser("template");s.add_argument("--packet",required=True);s.add_argument("--reviewer",required=True)
    s=sub.add_parser("reconcile");s.add_argument("--packet",required=True);s.add_argument("--reviews-a",required=True);s.add_argument("--reviews-b",required=True)
    s=sub.add_parser("failures");s.add_argument("--report",required=True);s.add_argument("--dataset",required=True);s.add_argument("--corpus",required=True)
    for s in sub.choices.values():s.add_argument("--output",required=True)
    a=p.parse_args()
    try:
        if a.command=="pool":result={"questions":len(pool(a.corpus,a.questions,a.runs,a.output)["questions"])}
        elif a.command=="template":template(a.packet,a.reviewer,a.output);result={"status":"blank_submission_not_completed"}
        elif a.command=="reconcile":result=reconcile(a.packet,a.reviews_a,a.reviews_b,a.output)
        else:result=failures(a.report,a.dataset,a.corpus,a.output)
        print(json.dumps(result))
    except (KeyError,ValueError,TypeError,OSError) as exc:p.exit(2,f"review refused: {exc}\n")

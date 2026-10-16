#!/usr/bin/env python3
"""Dependency-free, offline-first evidence review and retrieval evaluation.

No model/network calls unless the user explicitly supplies a retrieval adapter.
Legacy automatically sampled labels are proposals, never reviewed gold.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import time

MODES = ("vector", "hybrid", "hybrid_rerank")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(x, ensure_ascii=False, allow_nan=False) + "\n" for x in rows), encoding="utf-8", newline="\n")


def unique(rows, key):
    ids = [r[key] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate {key}")
    return {r[key]: r for r in rows}


def prepare(legacy, output, corpus=None, count=40):
    if not 30 <= count <= 50:
        raise ValueError("review batch must contain 30–50 questions")
    source = read_jsonl(legacy)
    if len(source) < count:
        raise ValueError("not enough source questions")
    chunks = unique(read_jsonl(corpus), "chunk_id") if corpus else {}
    corpus_hash = digest(corpus) if corpus else None
    rows = []
    for row in source[:count]:
        proposals = []
        for cid in row.get("ground_truth", []):
            chunk = chunks.get(cid)
            proposals.append({
                "chunk_id": cid,
                "content": chunk["content"] if chunk else None,
                "content_sha256": hashlib.sha256(chunk["content"].encode()).hexdigest() if chunk else None,
                "origin": "legacy_evenly_spaced_sampling_NOT_relevance_judgment",
                "relevance": None,
            })
        rows.append({
            "query_id": row["query_id"], "question": row["question"],
            "source_url": row.get("source_url"), "split": "development",
            "label_status": "pending_review", "reviewed_by": None, "reviewed_at": None,
            "corpus_sha256": corpus_hash,
            "candidate_evidence": proposals, "evidence": [],
            "review_notes": "Read full evidence, replace broad questions if needed, add/remove chunks across documents. No human approval has occurred.",
        })
    write_jsonl(output, rows)
    return {"questions": len(rows), "reviewed": 0, "corpus_present": bool(corpus)}


def validate(dataset, corpus, fixture=False):
    rows = read_jsonl(dataset)
    if not rows:
        raise ValueError("empty dataset")
    unique(rows, "query_id")
    chunks = unique(read_jsonl(corpus), "chunk_id")
    corpus_hash = digest(corpus)
    for row in rows:
        status = "fixture" if fixture else "reviewed"
        if row.get("label_status") != status:
            raise ValueError(f"{row['query_id']}: requires label_status={status}; candidates are not gold")
        if row.get("split") not in ("development", "test") or not row.get("question", "").strip():
            raise ValueError("missing split/question")
        if not fixture and (not row.get("reviewed_by") or not row.get("reviewed_at")):
            raise ValueError("reviewed gold requires reviewer and timestamp (user review, never auto-promote)")
        if row.get("corpus_sha256") != corpus_hash:
            raise ValueError("corpus fingerprint mismatch; review again after re-chunking")
        evidence = row.get("evidence", [])
        if not evidence:
            raise ValueError("answerable retrieval questions require at least one supporting evidence span")
        unique(evidence, "chunk_id")
        positives = 0
        for item in evidence:
            chunk = chunks.get(item["chunk_id"])
            if chunk is None:
                raise ValueError("evidence chunk is not in corpus")
            text = chunk["content"]
            start, end = item.get("start"), item.get("end")
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                raise ValueError("invalid Unicode evidence offsets")
            if text[start:end] != item.get("quote"):
                raise ValueError("quote does not match evidence span")
            if item.get("content_sha256") != hashlib.sha256(text.encode()).hexdigest():
                raise ValueError("evidence content fingerprint mismatch")
            if type(item.get("relevance")) is not int or not 0 <= item["relevance"] <= 3:
                raise ValueError("relevance must be integer 0–3")
            if item["relevance"] > 0:
                positives += 1
                if not item.get("rationale", "").strip():
                    raise ValueError("supporting evidence needs a rationale")
        if not positives:
            raise ValueError("no positive evidence; use future abstention suite for unanswerable questions")
    return rows, chunks


def metrics(ranked, relevant, k=5):
    if not relevant:
        raise ValueError("empty relevant set")
    if len(set(ranked)) != len(ranked):
        raise ValueError("duplicate result IDs would bias ranking")
    top = ranked[:k]
    hits = sorted(set(top) & set(relevant))
    reciprocal_rank = next((1 / i for i, cid in enumerate(top, 1) if cid in relevant), 0.0)
    return {"recall_at_5": len(hits) / len(relevant), "mrr_at_5": reciprocal_rank, "hit_ids": hits}


def p95(values):
    return sorted(values)[max(0, math.ceil(.95 * len(values)) - 1)]


def check_config(config, mode):
    required = {"revision", "embedding_model", "index_snapshot", "top_k", "candidate_k", "rrf_k", "rewrite", "kg", "reranker"}
    if not required <= config.keys() or config["top_k"] != 5:
        raise ValueError("incomplete run config or top_k != 5")
    if config["rewrite"] is not False or config["kg"] is not False:
        raise ValueError("phase-one comparison fixes rewrite=false and kg=false")
    if type(config["candidate_k"]) is not int or config["candidate_k"] < 5:
        raise ValueError("candidate_k must be >=5")
    if type(config["rrf_k"]) is not int or config["rrf_k"] <= 0:
        raise ValueError("rrf_k must be a positive integer")
    if any(not isinstance(config[key], str) or not config[key].strip() for key in ("revision", "embedding_model", "index_snapshot")):
        raise ValueError("revision, model and snapshot must be nonempty strings")
    if bool(config["reranker"]) != (mode == "hybrid_rerank"):
        raise ValueError("reranker config does not match mode")


def compare(dataset, corpus, paths, output, fixture=False):
    rows, chunks = validate(dataset, corpus, fixture)
    queries = unique(rows, "query_id")
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    if {x.get("mode") for x in runs} != set(MODES) or len(runs) != 3:
        raise ValueError("provide exactly vector, hybrid, hybrid_rerank runs")
    common = None
    report = {"kind": "OFFLINE_FIXTURE_NOT_REAL_RETRIEVAL" if fixture else "REVIEWED_RETRIEVAL_EXPORTS",
              "dataset_sha256": digest(dataset), "corpus_sha256": digest(corpus),
              "metrics_scope": "macro Recall@5 and truncated MRR@5; unjudged chunks count as nonrelevant",
              "modes": {}}
    for run in runs:
        if run.get("kind") != ("fixture" if fixture else "retrieval"):
            raise ValueError("fixture/real run mismatch")
        if run.get("dataset_sha256") != digest(dataset) or run.get("corpus_sha256") != digest(corpus):
            raise ValueError("run dataset/corpus fingerprint mismatch")
        config = run.get("config", {})
        check_config(config, run["mode"])
        shared = {k: v for k, v in config.items() if k != "reranker"}
        if common is None: common = shared
        if shared != common:
            raise ValueError("shared retrieval config changed between modes")
        predictions = unique(run.get("queries", []), "query_id")
        if predictions.keys() != queries.keys():
            raise ValueError("missing/extra query results; failed questions cannot be dropped")
        per_query = []
        for qid, row in queries.items():
            pred = predictions[qid]
            if "ranked_chunk_ids" not in pred:
                raise ValueError("each query must explicitly export ranking (empty list on error)")
            latency = pred.get("latency_ms")
            if type(latency) not in (int,float) or not math.isfinite(latency) or latency < 0:
                raise ValueError("latency_ms must be finite and nonnegative")
            ranked = pred.get("ranked_chunk_ids", [])
            if not isinstance(ranked, list) or any(cid not in chunks for cid in ranked):
                raise ValueError("unknown result chunk ID (export canonical child IDs)")
            if len(ranked) > 5:
                raise ValueError("export top 5 results only")
            if pred.get("error") and ranked:
                raise ValueError("failed queries cannot also export successful ranking")
            relevant = {e["chunk_id"] for e in row["evidence"] if e["relevance"] > 0}
            per_query.append({"query_id": qid, **metrics(ranked, relevant), "ranked_chunk_ids": ranked,
                              "latency_ms": latency, "error": pred.get("error"),
                              "usage": pred.get("usage"), "cost_usd": pred.get("cost_usd")})
        times = [p["latency_ms"] for p in per_query]
        report["modes"][run["mode"]] = {
            "config": config, "questions": len(per_query),
            "recall_at_5": statistics.mean(p["recall_at_5"] for p in per_query),
            "mrr_at_5": statistics.mean(p["mrr_at_5"] for p in per_query),
            "mean_latency_ms": statistics.mean(times), "p95_latency_ms": p95(times),
            "failed_queries": sum(bool(p["error"]) for p in per_query), "per_query": per_query,
        }
    write_json(output, report)
    return report


def collect(dataset, corpus, adapter_spec, config_path, output_dir):
    """Explicit opt-in adapter: retrieve(question, mode, config) -> result dict.

    Adapter must use canonical chunk IDs, fail instead of silently changing mode,
    and omit credentials from config/results. No adapter is executed by default.
    """
    rows, _ = validate(dataset, corpus)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    # Reject invalid plans before importing or executing any paid adapter.
    check_config(config, "hybrid_rerank")
    module, func = adapter_spec.split(":", 1)
    retrieve = getattr(importlib.import_module(module), func)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    runs = {mode: {"kind": "retrieval", "mode": mode, "dataset_sha256": digest(dataset),
                  "corpus_sha256": digest(corpus), "config": {**config, "reranker": config.get("reranker") if mode == "hybrid_rerank" else None},
                  "queries": []} for mode in MODES}
    # Rotate mode order by question to reduce systematic cache/order bias.
    for i, row in enumerate(rows):
        for mode in MODES[i % 3:] + MODES[:i % 3]:
            started = time.perf_counter()
            try:
                result = retrieve(row["question"], mode, dict(runs[mode]["config"]))
                if not isinstance(result, dict): raise ValueError("adapter must return a result object")
                result = {k: result[k] for k in ("ranked_chunk_ids", "error", "usage", "cost_usd") if k in result}
                if "ranked_chunk_ids" not in result and not result.get("error"):
                    raise ValueError("adapter did not return ranked_chunk_ids")
                if result.get("error"):
                    result.setdefault("ranked_chunk_ids", [])
            except Exception as exc:
                # Do not serialize exception text: SDK errors may expose credentials.
                result = {"ranked_chunk_ids": [], "error": type(exc).__name__}
            result.update(query_id=row["query_id"], latency_ms=(time.perf_counter() - started) * 1000)
            runs[mode]["queries"].append(result)
    paths = []
    for mode, run in runs.items():
        path = out / f"{mode}.json"; write_json(path, run); paths.append(path)
    return compare(dataset, corpus, paths, out / "report.json")


def fixture(output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    corpus = out / "corpus.jsonl"; dataset = out / "labels.jsonl"
    write_jsonl(corpus, [{"chunk_id": x, "content": f"Synthetic evidence {x}."} for x in ("a", "b", "c")])
    text = "Synthetic evidence a."
    write_jsonl(dataset, [{"query_id": "fixture-1", "question": "Synthetic metric check", "split": "development",
                          "label_status": "fixture", "corpus_sha256": digest(corpus),
                          "evidence": [{"chunk_id": "a", "start": 0, "end": len(text), "quote": text,
                                        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                        "relevance": 3, "rationale": "Synthetic arithmetic fixture only"}]}])
    paths = []
    for mode in MODES:
        path = out / f"{mode}.json"
        write_json(path, {"kind": "fixture", "mode": mode, "dataset_sha256": digest(dataset),
                          "corpus_sha256": digest(corpus), "config": {
                              "revision": "fixture", "embedding_model": "none", "index_snapshot": "synthetic",
                              "top_k": 5, "candidate_k": 20, "rrf_k": 60, "rewrite": False, "kg": False,
                              "reranker": "synthetic" if mode == "hybrid_rerank" else None},
                          "queries": [{"query_id": "fixture-1", "ranked_chunk_ids": ["b", "a", "c"], "latency_ms": 0}]})
        paths.append(path)
    return compare(dataset, corpus, paths, out / "report.json", fixture=True)


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare"); prep.add_argument("--legacy", required=True); prep.add_argument("--corpus")
    prep.add_argument("--output", required=True); prep.add_argument("--count", type=int, default=40)
    for name in ("validate", "compare", "collect"):
        s = sub.add_parser(name); s.add_argument("--dataset", required=True); s.add_argument("--corpus", required=True)
        if name == "compare": s.add_argument("--runs", nargs=3, required=True); s.add_argument("--output", required=True)
        if name == "collect":
            s.add_argument("--adapter", required=True); s.add_argument("--config", required=True); s.add_argument("--output-dir", required=True)
    f = sub.add_parser("fixture"); f.add_argument("--output-dir", required=True)
    a = p.parse_args()
    try:
        if a.command == "prepare": result = prepare(a.legacy, a.output, a.corpus, a.count)
        elif a.command == "validate": result = {"validated_questions": len(validate(a.dataset, a.corpus)[0])}
        elif a.command == "compare": result = compare(a.dataset, a.corpus, a.runs, a.output)
        elif a.command == "collect": result = collect(a.dataset, a.corpus, a.adapter, a.config, a.output_dir)
        else: result = fixture(a.output_dir)
        print(json.dumps({k: v for k, v in result.items() if k != "modes"}, ensure_ascii=False))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        p.exit(2, f"evaluation refused: {exc}\n")


if __name__ == "__main__":
    main()

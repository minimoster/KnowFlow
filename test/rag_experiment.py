#!/usr/bin/env python3
"""Archive existing RAG exports with reproducible comparisons; never call a model.

Scoring and evidence checks are delegated to rag_eval.compare. Hashes and a
recomputed report establish internal consistency, not who produced the inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import tempfile

import rag_eval as ev


VERSION = "rag_experiment_bundle_v1"
INPUTS = ("labels.jsonl", "corpus.jsonl", *(f"{mode}.json" for mode in ev.MODES))
ARTIFACTS = (*INPUTS, "report.json", "comparison.json", "REPORT.md")
FIXTURE = "OFFLINE_FIXTURE_NOT_REAL_RETRIEVAL"
REAL = "REVIEWED_RETRIEVAL_EXPORTS"
SCOPE = "Retrieval only: macro Recall@5 and truncated MRR@5; no answer-quality or end-to-end Agent claim."


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tool_hashes():
    return {"rag_eval.py": ev.digest(ev.__file__), "rag_experiment.py": ev.digest(__file__)}


def check_provenance(report, rows, fixture):
    splits = {row["split"] for row in rows}
    if len(splits) != 1:
        raise ValueError("archive development and test splits separately; do not pool them")
    for group in report["modes"].values():
        config = group["config"]
        if not fixture:
            if not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
                raise ValueError("real exports require a full 40-character tested revision SHA")
            for name in ("embedding_model", "index_snapshot", "reranker"):
                value = config[name]
                if value is not None and (not isinstance(value, str) or not value.strip()
                                         or "REPLACE_" in value.upper() or value.upper() == "TODO"):
                    raise ValueError("replace placeholder model, index and reranker identifiers before archiving")
        for row in group["per_query"]:
            error = row.get("error")
            if error is not None and not isinstance(error, str):
                raise ValueError("error must be a string or null")
            cost = row.get("cost_usd")
            if cost is not None and (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
                raise ValueError("cost_usd must be finite, nonnegative or null when unknown")
    return next(iter(splits))


def outcome(row):
    """Observable outcomes, not an inferred root cause of a failure."""
    if row.get("error"):
        return "execution_error"
    if not row["ranked_chunk_ids"]:
        return "empty_retrieval"
    if row["recall_at_5"] == 0:
        return "no_gold_hit"
    if row["recall_at_5"] < 1:
        return "partial_gold_hit"
    return "all_gold_hit"


def ranking_change(recall_delta, mrr_delta):
    if recall_delta == 0 and mrr_delta == 0:
        return "unchanged"
    if recall_delta >= 0 and mrr_delta >= 0:
        return "improved"
    if recall_delta <= 0 and mrr_delta <= 0:
        return "regressed"
    return "tradeoff"


def summarize(report, baseline):
    if baseline not in ev.MODES:
        raise ValueError("baseline must be an existing retrieval mode")
    groups = report["modes"]
    summary = {"kind": report["kind"], "baseline": baseline, "scope": SCOPE,
               "cost_scope": "Exported per-query retrieval cost only; missing cost is unknown, never zero.",
               "modes": {}, "comparisons": {}}
    reference = ev.unique(groups[baseline]["per_query"], "query_id")
    for mode in ev.MODES:
        group = groups[mode]
        cases = group["per_query"]
        known = [row["cost_usd"] for row in cases if row.get("cost_usd") is not None]
        try:
            subtotal = math.fsum(known) if known else None
        except OverflowError as exc:
            raise ValueError("cost_usd total exceeds finite numeric range") from exc
        buckets = {name: [] for name in ("execution_error", "empty_retrieval", "no_gold_hit",
                                        "partial_gold_hit", "all_gold_hit")}
        for row in cases:
            buckets[outcome(row)].append(row["query_id"])
        summary["modes"][mode] = {
            **{key: group[key] for key in ("questions", "recall_at_5", "mrr_at_5", "mean_latency_ms",
                                           "p95_latency_ms", "failed_queries")},
            "cost": {"known_queries": len(known), "unknown_queries": len(cases) - len(known),
                     "coverage": len(known) / len(cases),
                     "known_subtotal_usd": subtotal,
                     "total_usd": subtotal if len(known) == len(cases) else None},
            "outcome_query_ids": buckets,
        }
        if mode == baseline:
            continue
        paired = []
        for candidate in cases:
            before = reference[candidate["query_id"]]
            recall_delta = candidate["recall_at_5"] - before["recall_at_5"]
            mrr_delta = candidate["mrr_at_5"] - before["mrr_at_5"]
            before_error, after_error = bool(before.get("error")), bool(candidate.get("error"))
            transition = ("both_failed" if before_error and after_error else
                          "recovered_error" if before_error else "new_error" if after_error else "both_succeeded")
            before_cost, after_cost = before.get("cost_usd"), candidate.get("cost_usd")
            paired.append({"query_id": candidate["query_id"],
                           "recall_at_5_delta": recall_delta, "mrr_at_5_delta": mrr_delta,
                           "latency_ms_delta": candidate["latency_ms"] - before["latency_ms"],
                           "cost_usd_delta": after_cost - before_cost if before_cost is not None and after_cost is not None else None,
                           "ranking_change": ranking_change(recall_delta, mrr_delta),
                           "execution_transition": transition,
                           "baseline_outcome": outcome(before), "candidate_outcome": outcome(candidate)})
        summary["comparisons"][mode] = {
            "paired_questions": len(paired),
            "recall_at_5_delta": group["recall_at_5"] - groups[baseline]["recall_at_5"],
            "mrr_at_5_delta": group["mrr_at_5"] - groups[baseline]["mrr_at_5"],
            "mean_latency_ms_delta": group["mean_latency_ms"] - groups[baseline]["mean_latency_ms"],
            "ranking_changes": dict(Counter(row["ranking_change"] for row in paired)),
            "execution_transitions": dict(Counter(row["execution_transition"] for row in paired)),
            "per_query": paired,
        }
    return summary


def cell(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def markdown(report, summary, split):
    lines = ["# RAG 检索实验对照", "", f"**结果类别：{report['kind']}**", "",
             "本页只汇总已有导出，不连接模型或数据库。fixture 仅验证工具；真实导出也不自动证明输入真实性。", "",
             f"基线：`{summary['baseline']}`；数据切分：`{split}`。三组使用相同语料、题目和共有配置。", "",
             "范围：有答案题的宏平均 Recall@5、截断 MRR@5。不是答案质量、拒答能力或 Agent 总体成功率。", "",
             f"数据 SHA-256：`{report['dataset_sha256']}`", "",
             f"语料 SHA-256：`{report['corpus_sha256']}`", "",
             "## 三组结果", "",
             "| 组别 | 题数 | Recall@5 | MRR@5 | 平均 / P95 毫秒 | 执行错误 | 成本已知题数 | 总成本 USD |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mode in ev.MODES:
        group = summary["modes"][mode]
        cost = group["cost"]
        total = f"{cost['total_usd']:.6f}" if cost["total_usd"] is not None else "未知（未完整采集）"
        lines.append(f"| {mode} | {group['questions']} | {group['recall_at_5']:.4f} | {group['mrr_at_5']:.4f} | "
                     f"{group['mean_latency_ms']:.3f} / {group['p95_latency_ms']:.3f} | {group['failed_queries']} | "
                     f"{cost['known_queries']}/{group['questions']} | {total} |")
    lines += ["", "成本未采集时保持未知；部分已知金额只是小计，见 comparison.json，不当作总成本。耗时沿用导出器口径，"
              "涵盖其检索调用，不包含数据入库及最终答案生成。", "", "## 相对基线的配对变化", "",
              "| 候选 | ΔRecall 百分点 | ΔMRR | Δ平均毫秒 | 改善 / 退步 / 权衡 / 不变 |",
              "| --- | ---: | ---: | ---: | --- |"]
    for mode, comparison in summary["comparisons"].items():
        counts = comparison["ranking_changes"]
        tally = " / ".join(str(counts.get(name, 0)) for name in ("improved", "regressed", "tradeoff", "unchanged"))
        lines.append(f"| {mode} | {comparison['recall_at_5_delta'] * 100:+.3f} | {comparison['mrr_at_5_delta']:+.4f} | "
                     f"{comparison['mean_latency_ms_delta']:+.3f} | {tally} |")
    lines += ["", "改善/退步要求 Recall 与 MRR 同向或一项不变；一升一降记为权衡。错误恢复单独记录，"
              "不会因为同为零分而丢失。小样本和单次运行的差值不代表统计显著性或稳定收益。", "",
              "## 待复盘题目", "", "下面是可观察的结果类别，不是自动判定的根因；全部题目保留在 report.json。", "",
              "| 组别 | 执行错误 | 空检索 | 未命中已标证据 | 部分命中已标证据 |",
              "| --- | --- | --- | --- | --- |"]
    for mode in ev.MODES:
        buckets = summary["modes"][mode]["outcome_query_ids"]
        values = [cell(", ".join(map(str, buckets[name]))) or "无" for name in
                  ("execution_error", "empty_retrieval", "no_gold_hit", "partial_gold_hit")]
        lines.append("| " + " | ".join([mode, *values]) + " |")
    lines += ["", "## 复现边界", "", "- 执行错误按零分计入全体题目，不能删除失败题提高均值。",
              "- 未标注候选按不相关处理，证据标注不完整会影响结果。",
              "- manifest.json 记录文件及评分脚本哈希；verify 重算评分与报告，检查归档内部一致性。",
              "- 哈希不提供来源认证，不能证明模型确实运行、审阅者为真人或远端索引未变化。",
              "- 原始逐题导出、配置与用量字段保持在各组 JSON 中。本报告不复核未提供的历史 SciFact 实验。", ""]
    return "\n".join(lines)


def analyze(directory, baseline, fixture):
    root = Path(directory)
    rows, _ = ev.validate(root / "labels.jsonl", root / "corpus.jsonl", fixture)
    report = ev.compare(root / "labels.jsonl", root / "corpus.jsonl",
                        [root / f"{mode}.json" for mode in ev.MODES], root / "report.json", fixture)
    split = check_provenance(report, rows, fixture)
    summary = summarize(report, baseline)
    ev.write_json(root / "comparison.json", summary)
    (root / "REPORT.md").write_text(markdown(report, summary, split), encoding="utf-8", newline="\n")
    return report, split


def pack(dataset, corpus, runs, output_dir, baseline="vector", fixture=False):
    """Read inputs once; validate staged copies before creating a new output directory."""
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError("output directory already exists; choose a new experiment directory")
    if baseline not in ev.MODES:
        raise ValueError("unknown baseline mode")
    inputs = {"labels.jsonl": Path(dataset).read_bytes(), "corpus.jsonl": Path(corpus).read_bytes()}
    if len(runs) != len(ev.MODES):
        raise ValueError("provide exactly three existing raw exports")
    for path in runs:
        data = Path(path).read_bytes()
        mode = json.loads(data).get("mode")
        name = f"{mode}.json"
        if mode not in ev.MODES or name in inputs:
            raise ValueError("provide each retrieval mode exactly once")
        inputs[name] = data
    with tempfile.TemporaryDirectory(prefix="knowflow-rag-pack-") as temp:
        staging = Path(temp)
        for name, data in inputs.items():
            (staging / name).write_bytes(data)
        report, split = analyze(staging, baseline, fixture)
        manifest = {"schema": VERSION, "classification": report["kind"], "baseline": baseline,
                    "split": split, "questions": report["modes"][baseline]["questions"],
                    "retrieval_revision": report["modes"][baseline]["config"]["revision"],
                    "scope": SCOPE, "tools_sha256": tool_hashes(),
                    "files": {name: {"sha256": ev.digest(staging / name), "bytes": (staging / name).stat().st_size}
                              for name in ARTIFACTS}}
        ev.write_json(staging / "manifest.json", manifest)
        # mkdir's exclusive final component protects existing experiment directories.
        # If a subsequent filesystem write fails, verify will reject the incomplete bundle.
        out.mkdir(parents=True, exist_ok=False)
        for name in (*ARTIFACTS, "manifest.json"):
            with (out / name).open("xb") as target:
                target.write((staging / name).read_bytes())
    return manifest


def verify(directory):
    root = Path(directory)
    expected = set(ARTIFACTS) | {"manifest.json"}
    if {path.name for path in root.iterdir()} != expected:
        raise ValueError("bundle must contain exactly the declared experiment files")
    if any((root / name).is_symlink() or not (root / name).is_file() for name in expected):
        raise ValueError("bundle members must be regular files, not symbolic links")
    manifest = load_json(root / "manifest.json")
    if manifest.get("schema") != VERSION or manifest.get("classification") not in (REAL, FIXTURE):
        raise ValueError("unsupported experiment manifest")
    if manifest.get("baseline") not in ev.MODES or manifest.get("scope") != SCOPE:
        raise ValueError("invalid baseline or metric scope")
    if manifest.get("tools_sha256") != tool_hashes():
        raise ValueError("scoring scripts changed; use the versions whose hashes were archived")
    if set(manifest.get("files", {})) != set(ARTIFACTS):
        raise ValueError("manifest file set differs")
    for name in ARTIFACTS:
        entry = manifest["files"][name]
        if entry != {"sha256": ev.digest(root / name), "bytes": (root / name).stat().st_size}:
            raise ValueError("archived file fingerprint differs: " + name)
    with tempfile.TemporaryDirectory(prefix="knowflow-rag-verify-") as temp:
        staging = Path(temp)
        for name in INPUTS:
            (staging / name).write_bytes((root / name).read_bytes())
        report, split = analyze(staging, manifest["baseline"], manifest["classification"] == FIXTURE)
        for name in ("report.json", "comparison.json", "REPORT.md"):
            if (staging / name).read_bytes() != (root / name).read_bytes():
                raise ValueError("stored report differs from recomputed raw exports: " + name)
        group = report["modes"][manifest["baseline"]]
        if (manifest.get("classification") != report["kind"] or manifest.get("split") != split
                or manifest.get("questions") != group["questions"]
                or manifest.get("retrieval_revision") != group["config"]["revision"]):
            raise ValueError("manifest metadata differs from raw exports")
    return {"schema": VERSION, "classification": manifest["classification"], "questions": manifest["questions"],
            "baseline": manifest["baseline"], "integrity": "consistent", "scope": SCOPE}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("pack", help="archive and compare existing exports; no adapter execution")
    create.add_argument("--dataset", required=True)
    create.add_argument("--corpus", required=True)
    create.add_argument("--runs", nargs=3, required=True)
    create.add_argument("--output-dir", required=True)
    create.add_argument("--baseline", choices=ev.MODES, default="vector")
    create.add_argument("--fixture", action="store_true", help="synthetic arithmetic inputs only, prominently labeled")
    check = sub.add_parser("verify", help="check hashes and recompute archived reports")
    check.add_argument("--bundle", required=True)
    args = parser.parse_args()
    try:
        if args.command == "pack":
            result = pack(args.dataset, args.corpus, args.runs, args.output_dir, args.baseline, args.fixture)
            result = {key: result[key] for key in ("schema", "classification", "questions", "baseline", "scope")}
        else:
            result = verify(args.bundle)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        parser.exit(2, f"experiment refused: {exc}\n")


if __name__ == "__main__":
    main()

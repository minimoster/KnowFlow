#!/usr/bin/env python3
"""Offline corpus/snapshot binding and pending evidence packets. Never invent IDs or labels."""
import argparse
import hashlib
import json
from pathlib import Path

import rag_eval as ev


def content_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def identity(row):
    doc, idx = row.get('doc_hash'), row.get('chunk_idx')
    if not isinstance(doc, str) or not doc.strip() or type(idx) is not int or idx < 0:
        raise ValueError('actual ingestion doc_hash and chunk_idx required; never infer from legacy chunk_index')
    return doc, idx


def bind(corpus, snapshot):
    if not corpus or not snapshot:
        raise ValueError('empty corpus/snapshot is not a real evaluation dataset')
    chunks = ev.unique(corpus, 'chunk_id')
    by_key = {}
    for cid, row in chunks.items():
        if not isinstance(cid, str) or not cid.strip():
            raise ValueError('nonempty canonical chunk_id required')
        text = row.get('content')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('original child content missing')
        if not isinstance(row.get('source_url'), str) or not row['source_url'].startswith(('https://', 'http://')):
            raise ValueError('original source_url required')
        if row.get('content_sha256', content_hash(text)) != content_hash(text):
            raise ValueError('corpus content fingerprint differs')
        key = identity(row)
        if key in by_key:
            raise ValueError('duplicate ingestion identity; content alone cannot disambiguate')
        by_key[key] = row
    seen_ids, seen_keys, mapping = set(), set(), []
    for row in snapshot:
        pgid = row.get('pg_id')
        if type(pgid) is not int or pgid <= 0 or pgid in seen_ids:
            raise ValueError('PG IDs must be unique positive integers')
        key = identity(row)
        if key in seen_keys or key not in by_key:
            raise ValueError('duplicate or out-of-corpus PG row; isolate the evaluation index')
        chunk = by_key[key]
        if row.get('content') != chunk['content']:
            raise ValueError('PG child content differs from frozen source')
        seen_ids.add(pgid); seen_keys.add(key)
        mapping.append({'pg_id': pgid, 'chunk_id': chunk['chunk_id'], 'doc_hash': key[0],
                        'chunk_idx': key[1], 'content_sha256': content_hash(chunk['content'])})
    if seen_keys != set(by_key):
        raise ValueError('missing PG rows for corpus chunks')
    return sorted(mapping, key=lambda r: r['pg_id'])


def freeze(corpus_path, snapshot_path, index_snapshot, output):
    if not index_snapshot.strip():
        raise ValueError('operator-supplied index snapshot identifier required')
    corpus, snapshot = ev.read_jsonl(corpus_path), ev.read_jsonl(snapshot_path)
    mapping = bind(corpus, snapshot)  # Validate all inputs before creating output.
    corpus = sorted([{**r, 'content_sha256': content_hash(r['content'])} for r in corpus], key=lambda r: r['chunk_id'])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite a frozen dataset.
    ev.write_jsonl(out / 'corpus.jsonl', corpus)
    ev.write_jsonl(out / 'pg_snapshot.jsonl', sorted(snapshot, key=lambda r: r['pg_id']))
    ev.write_jsonl(out / 'mapping.jsonl', mapping)
    manifest = {'kind': 'frozen_rag_corpus_v1', 'index_snapshot': index_snapshot,
                'chunk_count': len(corpus), 'mapping_count': len(mapping),
                'corpus_sha256': ev.digest(out / 'corpus.jsonl'),
                'mapping_sha256': ev.digest(out / 'mapping.jsonl'),
                'pg_snapshot_sha256': ev.digest(out / 'pg_snapshot.jsonl'),
                'input_corpus_sha256': ev.digest(corpus_path), 'input_pg_export_sha256': ev.digest(snapshot_path),
                'scope': 'Exact PG child rows bound by ingestion identity and content; ES/Milvus snapshot identity is operator supplied, not remotely attested.'}
    ev.write_json(out / 'manifest.json', manifest)
    return manifest


def validate_bundle(directory):
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('kind') != 'frozen_rag_corpus_v1' or not manifest.get('index_snapshot'):
        raise ValueError('invalid frozen corpus manifest')
    for file, field in [('corpus.jsonl', 'corpus_sha256'), ('mapping.jsonl', 'mapping_sha256'), ('pg_snapshot.jsonl', 'pg_snapshot_sha256')]:
        if ev.digest(root / file) != manifest.get(field):
            raise ValueError('frozen file fingerprint differs: ' + file)
    corpus, snapshot = ev.read_jsonl(root / 'corpus.jsonl'), ev.read_jsonl(root / 'pg_snapshot.jsonl')
    mapping = ev.read_jsonl(root / 'mapping.jsonl')
    if mapping != bind(corpus, snapshot) or len(corpus) != manifest.get('chunk_count') or len(mapping) != manifest.get('mapping_count'):
        raise ValueError('frozen mapping/count differs')
    return manifest, corpus, mapping


def review_materials(questions, output, bundle=None):
    rows = ev.read_jsonl(questions); ev.unique(rows, 'query_id')
    manifest, corpus, _ = validate_bundle(bundle) if bundle else ({}, [], [])
    chunks = ev.unique(corpus, 'chunk_id')
    packets, missing = [], set()
    for row in rows:
        candidates = []
        for candidate in row.get('candidate_evidence', []):
            cid = candidate['chunk_id']; chunk = chunks.get(cid)
            if chunk is None: missing.add(cid)
            candidates.append({'chunk_id': cid, 'availability': 'available_unreviewed' if chunk else 'missing_source',
                               'content': chunk['content'] if chunk else None,
                               'content_sha256': content_hash(chunk['content']) if chunk else None,
                               'source_url': chunk['source_url'] if chunk else row.get('source_url'),
                               'relevance': None, 'origin': 'legacy_candidate_NOT_relevance_judgment'})
        packets.append({'query_id': row['query_id'], 'question': row['question'], 'source_url': row.get('source_url'),
                        'split': row.get('split', 'development'), 'label_status': 'pending_review',
                        'corpus_sha256': manifest.get('corpus_sha256'), 'reviewed_by': None, 'reviewed_at': None,
                        'candidate_evidence': candidates, 'evidence': [],
                        'review_notes': 'Read original evidence; no automatic or human relevance judgment supplied. Pool all three arms before final review.'})
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    ev.write_jsonl(out / 'questions.jsonl', packets)
    report = {'kind': 'EVIDENCE_READINESS_NOT_RAG_RESULTS', 'questions_sha256': ev.digest(questions),
              'questions': len(rows), 'human_reviewed': 0, 'pending_review': len(rows),
              'available_candidate_occurrences': sum(c['availability'] == 'available_unreviewed' for r in packets for c in r['candidate_evidence']),
              'missing_candidate_ids': sorted(missing), 'corpus_supplied': bundle is not None,
              'required_materials': ['Frozen original child corpus', 'Actual ingestion identity receipts (doc_hash, chunk_idx)',
                                     'Read-only PG snapshot and immutable ES/Milvus evaluation indexes', 'Independent evidence judgments']}
    ev.write_json(out / 'readiness.json', report)
    lines = ['# 40题原文证据待审材料', '', '本文件不包含人工审定结论；缺失原文不得用项目文档或生成文本替代。', '']
    for row in packets:
        lines += [f"## {row['query_id']} — {row['question']}", '', f"原始来源：{row['source_url']}", '',
                  '状态：pending_review；审阅人、支持引文、等级和理由均待填写。', '',
                  *[f"- `{c['chunk_id']}`：{c['availability']}" for c in row['candidate_evidence']], '']
    (out / 'REVIEW.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('freeze'); s.add_argument('--corpus', required=True); s.add_argument('--pg-export', required=True)
    s.add_argument('--index-snapshot', required=True); s.add_argument('--output-dir', required=True)
    s = sub.add_parser('validate'); s.add_argument('--bundle', required=True)
    s = sub.add_parser('review'); s.add_argument('--questions', required=True); s.add_argument('--bundle'); s.add_argument('--output-dir', required=True)
    a = p.parse_args()
    try:
        if a.command == 'freeze': result = freeze(a.corpus, a.pg_export, a.index_snapshot, a.output_dir)
        elif a.command == 'validate': result = validate_bundle(a.bundle)[0]
        else: result = review_materials(a.questions, a.output_dir, a.bundle)
        print(json.dumps(result, ensure_ascii=False))
    except (OSError, KeyError, ValueError, TypeError) as exc: p.exit(2, f'data preparation refused: {exc}\n')


if __name__ == '__main__': main()

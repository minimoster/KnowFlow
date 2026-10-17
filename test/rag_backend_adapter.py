#!/usr/bin/env python3
"""Opt-in subprocess bridge to cmd/rag-export; pending-question export never scores labels."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

import rag_data as data
import rag_eval as ev


class BackendAdapter:
    def __init__(self, bundle, executable, backend_config, allow_model_calls=False):
        self.root = Path(bundle)
        self.manifest, _, mapping = data.validate_bundle(bundle)
        self.mapping = {r['pg_id']: r for r in mapping}
        self.executable = str(Path(executable).resolve())
        self.backend_config = str(Path(backend_config).resolve())
        self.allow_model_calls = allow_model_calls

    def validate_config(self, config, mode):
        ev.check_config(config, mode)
        expected = {key: self.manifest[key] for key in ('index_snapshot', 'corpus_sha256', 'mapping_sha256', 'pg_snapshot_sha256')}
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError('experiment is not bound to this frozen corpus/mapping/index snapshot')
        if config.get('exporter_sha256') != ev.digest(self.executable):
            raise ValueError('exporter binary fingerprint differs')
        if config.get('backend_config_sha256') != ev.digest(self.backend_config):
            raise ValueError('backend settings fingerprint differs')
        if type(config.get('rerank_preview_len')) is not int or config['rerank_preview_len'] <= 0:
            raise ValueError('actual rerank preview length required across arms')
        for filename, field in [('corpus.jsonl', 'corpus_sha256'), ('mapping.jsonl', 'mapping_sha256'), ('pg_snapshot.jsonl', 'pg_snapshot_sha256')]:
            if ev.digest(self.root / filename) != self.manifest[field]:
                raise ValueError('frozen bundle changed during collection')

    def decode(self, payload, question, mode, config):
        if payload.get('kind') != 'knowflow_backend_export_v1' or payload.get('mode') != mode or payload.get('question_sha256') != data.content_hash(question):
            raise ValueError('backend response identity mismatch')
        actual = payload.get('actual_config', {})
        for key in ('revision', 'embedding_model', 'reranker', 'rerank_preview_len', 'candidate_k', 'rrf_k', 'top_k', 'rewrite', 'kg'):
            if key not in actual or actual[key] != config.get(key):
                raise ValueError('backend configuration differs: ' + key)
        latency = payload.get('retrieval_latency_ms')
        if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
            raise ValueError('invalid backend latency')
        hits = payload.get('hits')
        if not isinstance(hits, list) or len(hits) > 5:
            raise ValueError('invalid backend hits')
        ranked, seen = [], set()
        for hit in hits:
            pgid = hit.get('pg_id')
            if type(pgid) is not int or pgid in seen or pgid not in self.mapping:
                raise ValueError('duplicate/unknown PGID; never guess from content')
            row = self.mapping[pgid]
            if hit.get('doc_hash') != row['doc_hash'] or type(hit.get('chunk_idx')) is not int or hit['chunk_idx'] != row['chunk_idx']:
                raise ValueError('live ingestion identity differs from frozen mapping')
            if hit.get('content_sha256') != row['content_sha256']:
                raise ValueError('live child content differs from frozen mapping')
            seen.add(pgid); ranked.append(row['chunk_id'])
        return {'ranked_chunk_ids': ranked, 'backend_latency_ms': latency}

    def retrieve(self, question, mode, config):
        # All local provenance checks precede subprocess creation or any model call.
        self.validate_config(config, mode)
        if not self.allow_model_calls:
            raise ValueError('model/backend execution has not been explicitly enabled')
        request = {'question': question, 'mode': mode, 'candidate_k': config['candidate_k'], 'rrf_k': config['rrf_k'],
                   'expected_config': {k:config[k] for k in ('revision', 'embedding_model', 'reranker', 'rerank_preview_len')}}
        proc = subprocess.run([self.executable, '--backend-config', self.backend_config,
                               '--allow-backend-read', '--allow-model-calls', '--timeout', '45s'],
                              input=json.dumps(request), capture_output=True, text=True, encoding='utf-8',
                              timeout=60, shell=False)
        if proc.returncode != 0:
            raise RuntimeError('backend_export_failed')  # No SDK stderr or tokens in persisted output.
        return self.decode(json.loads(proc.stdout), question, mode, config)


def retrieve(question, mode, config):
    """rag_eval.collect-compatible opt-in adapter; no I/O or model work at import time."""
    if os.environ.get('KNOWFLOW_EVAL_ALLOW_MODEL_CALLS') != '1':
        raise ValueError('explicit model-call opt-in required')
    adapter = BackendAdapter(os.environ['KNOWFLOW_EVAL_BUNDLE'], os.environ['KNOWFLOW_EVAL_EXPORTER'],
                             os.environ['KNOWFLOW_EVAL_BACKEND_CONFIG'], allow_model_calls=True)
    return adapter.retrieve(question, mode, config)


def export_questions(questions, config_path, adapter, output):
    rows = ev.read_jsonl(questions); ev.unique(rows, 'query_id')
    if not rows or any(not isinstance(r.get('question'), str) or not r['question'].strip() for r in rows):
        raise ValueError('nonempty question set required')
    if any(r.get('corpus_sha256') != adapter.manifest['corpus_sha256'] for r in rows):
        raise ValueError('questions must be bound to frozen corpus before export')
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    adapter.validate_config(config, 'hybrid_rerank')
    if not adapter.allow_model_calls:
        raise ValueError('explicit execution opt-in required')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    runs = {m: {'kind': 'retrieval', 'mode': m, 'dataset_sha256': ev.digest(questions),
                'corpus_sha256': adapter.manifest['corpus_sha256'],
                'config': {**config, 'reranker': config['reranker'] if m == 'hybrid_rerank' else None},
                'label_scope': 'Unscored raw retrieval; no human label assertion', 'queries': []} for m in ev.MODES}
    for i, row in enumerate(rows):
        for mode in ev.MODES[i % 3:] + ev.MODES[:i % 3]:
            started = time.perf_counter()
            try:
                result = adapter.retrieve(row['question'], mode, runs[mode]['config'])
            except Exception as exc:
                result = {'ranked_chunk_ids': [], 'error': type(exc).__name__}
            result.update(query_id=row['query_id'], latency_ms=(time.perf_counter() - started) * 1000)
            runs[mode]['queries'].append(result)
    for mode, run in runs.items(): ev.write_json(out / (mode + '.json'), run)
    return {'kind': 'UNSCORED_BACKEND_EXPORTS', 'questions': len(rows),
            'failed_calls': sum(bool(q.get('error')) for r in runs.values() for q in r['queries']),
            'note': 'Pool candidates, independently review evidence, then rerun using the reviewed dataset hash before scoring.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('questions', 'config', 'bundle', 'executable', 'backend-config', 'output-dir'): p.add_argument('--' + key, required=True)
    p.add_argument('--allow-model-calls', action='store_true')
    args = p.parse_args()
    try:
        adapter = BackendAdapter(args.bundle, args.executable, args.backend_config, args.allow_model_calls)
        print(json.dumps(export_questions(args.questions, args.config, adapter, args.output_dir)))
    except (OSError, KeyError, ValueError, TypeError) as exc: p.exit(2, f'export refused: {type(exc).__name__}\n')

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import rag_data as data
import rag_eval as ev
import rag_review
from rag_backend_adapter import BackendAdapter, export_questions


class DataChainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # Synthetic data only: these never enter the checked-in real review queue.
        self.corpus = [{'chunk_id': 'canonical-a', 'content': 'same text', 'source_url': 'https://example.invalid/a', 'doc_hash': 'doc-a', 'chunk_idx': 2},
                       {'chunk_id': 'canonical-b', 'content': 'same text', 'source_url': 'https://example.invalid/b', 'doc_hash': 'doc-b', 'chunk_idx': 2}]
        self.snapshot = [{'pg_id': 10, 'doc_hash': 'doc-a', 'chunk_idx': 2, 'content': 'same text'},
                         {'pg_id': 20, 'doc_hash': 'doc-b', 'chunk_idx': 2, 'content': 'same text'}]
        ev.write_jsonl(self.root/'input.jsonl', self.corpus); ev.write_jsonl(self.root/'pg.jsonl', self.snapshot)
        self.bundle = self.root/'frozen'
        self.manifest = data.freeze(self.root/'input.jsonl', self.root/'pg.jsonl', 'synthetic-index', self.bundle)
        self.exe = self.root/'dummy-executable'; self.exe.write_text('never executed')
        self.settings = self.root/'settings.json'; self.settings.write_text('{}')
        self.adapter = BackendAdapter(self.bundle, self.exe, self.settings)
        self.config = {key: self.manifest[key] for key in ('index_snapshot', 'corpus_sha256', 'mapping_sha256', 'pg_snapshot_sha256')}
        self.config.update(revision='test-source', embedding_model='synthetic', reranker='synthetic', rerank_preview_len=200,
                           candidate_k=20, rrf_k=60, top_k=5, rewrite=False, kg=False,
                           exporter_sha256=ev.digest(self.exe), backend_config_sha256=ev.digest(self.settings))

    def payload(self, mode='hybrid_rerank'):
        return {'kind': 'knowflow_backend_export_v1', 'mode': mode, 'question_sha256': data.content_hash('q'),
                'actual_config': {k:self.config[k] for k in ('revision','embedding_model','reranker','rerank_preview_len','candidate_k','rrf_k','top_k','rewrite','kg')},
                'hits': [{'pg_id':20,'doc_hash':'doc-b','chunk_idx':2,'content_sha256':data.content_hash('same text')}], 'retrieval_latency_ms':1.5}

    def test_freeze_deterministic_and_never_overwrites(self):
        ev.write_jsonl(self.root/'reverse.jsonl', list(reversed(self.corpus)))
        other = self.root/'other'
        data.freeze(self.root/'reverse.jsonl', self.root/'pg.jsonl', 'synthetic-index', other)
        self.assertEqual(ev.digest(other/'corpus.jsonl'), self.manifest['corpus_sha256'])
        with self.assertRaises(FileExistsError): data.freeze(self.root/'input.jsonl', self.root/'pg.jsonl', 'synthetic-index', self.bundle)

    def test_identical_content_different_documents_stays_distinct(self):
        self.assertEqual([r['chunk_id'] for r in data.bind(self.corpus,self.snapshot)], ['canonical-a','canonical-b'])

    def test_missing_duplicate_unknown_and_changed_pg_rows_refused(self):
        variants = [self.snapshot[:1], self.snapshot+[self.snapshot[0]], [dict(self.snapshot[0],pg_id=True),self.snapshot[1]],
                    [dict(self.snapshot[0],doc_hash='unknown'),self.snapshot[1]], [dict(self.snapshot[0],content='parent'),self.snapshot[1]]]
        for rows in variants:
            with self.subTest(rows=rows), self.assertRaises(ValueError): data.bind(self.corpus, rows)

    def test_legacy_identity_not_guessed(self):
        rows = copy.deepcopy(self.corpus); rows[0]['chunk_index']=rows[0].pop('chunk_idx')
        with self.assertRaises(ValueError): data.bind(rows,self.snapshot)

    def test_frozen_mapping_tamper_refused_even_with_updated_file_hash(self):
        rows=ev.read_jsonl(self.bundle/'mapping.jsonl'); rows[0]['chunk_id']='canonical-b'
        ev.write_jsonl(self.bundle/'mapping.jsonl',rows)
        manifest=dict(self.manifest,mapping_sha256=ev.digest(self.bundle/'mapping.jsonl'))
        ev.write_json(self.bundle/'manifest.json',manifest)
        with self.assertRaises(ValueError): data.validate_bundle(self.bundle)

    def test_missing_sources_stay_pending_and_empty(self):
        q=self.root/'q.jsonl'
        ev.write_jsonl(q,[{'query_id':'q','question':'q','source_url':'https://example.invalid','candidate_evidence':[{'chunk_id':'absent'}]}])
        report=data.review_materials(q,self.root/'review')
        self.assertEqual(report['human_reviewed'],0); self.assertEqual(report['missing_candidate_ids'],['absent'])
        row=ev.read_jsonl(self.root/'review/questions.jsonl')[0]
        self.assertEqual(row['label_status'],'pending_review'); self.assertIsNone(row['candidate_evidence'][0]['content'])

    def test_backend_pg_id_mapping_uses_child_fingerprint(self):
        result=self.adapter.decode(self.payload(),'q','hybrid_rerank',self.config)
        self.assertEqual(result['ranked_chunk_ids'],['canonical-b'])
        for hit in [{'pg_id':99,'content_sha256':data.content_hash('same text')}, {'pg_id':20,'doc_hash':'doc-b','chunk_idx':2,'content_sha256':'parent-hash'}, {'pg_id':True}]:
            payload=self.payload();payload['hits']=[hit]
            with self.assertRaises(ValueError): self.adapter.decode(payload,'q','hybrid_rerank',self.config)

    def test_backend_mode_question_config_and_duplicates_refused(self):
        variants=[]
        for key,value in [('mode','vector'),('question_sha256','wrong'),('retrieval_latency_ms',float('nan'))]:
            payload=self.payload();payload[key]=value;variants.append(payload)
        payload=self.payload();payload['hits']*=2;variants.append(payload)
        payload=self.payload();payload['actual_config']['revision']='other-source';variants.append(payload)
        payload=self.payload();payload['hits'][0]['doc_hash']='doc-a';variants.append(payload)
        for payload in variants:
            with self.assertRaises(ValueError): self.adapter.decode(payload,'q','hybrid_rerank',self.config)

    def test_missing_opt_in_and_binary_drift_never_spawn(self):
        with patch('rag_backend_adapter.subprocess.run') as run:
            with self.assertRaises(ValueError): self.adapter.retrieve('q','hybrid_rerank',self.config)
            self.adapter.allow_model_calls=True;self.exe.write_text('changed')
            with self.assertRaises(ValueError): self.adapter.retrieve('q','hybrid_rerank',self.config)
            run.assert_not_called()

    def test_bridge_invocation_and_redacted_error(self):
        self.adapter.allow_model_calls=True
        with patch('rag_backend_adapter.subprocess.run',return_value=subprocess.CompletedProcess([],0,json.dumps(self.payload()),'')) as run:
            self.assertEqual(self.adapter.retrieve('q','hybrid_rerank',self.config)['ranked_chunk_ids'],['canonical-b'])
            self.assertFalse(run.call_args.kwargs['shell']);self.assertEqual(run.call_args.kwargs['timeout'],60)
        with patch('rag_backend_adapter.subprocess.run',return_value=subprocess.CompletedProcess([],1,'','secret-response-body')):
            with self.assertRaisesRegex(RuntimeError,'^backend_export_failed$'): self.adapter.retrieve('q','hybrid_rerank',self.config)

    def test_timeout_does_not_return_cached_or_mock_success(self):
        self.adapter.allow_model_calls=True
        with patch('rag_backend_adapter.subprocess.run',side_effect=subprocess.TimeoutExpired('dummy',60)):
            with self.assertRaises(subprocess.TimeoutExpired): self.adapter.retrieve('q','hybrid_rerank',self.config)

    def test_pending_export_to_blind_pool_keeps_labels_unreviewed(self):
        q=self.root/'q.jsonl'; cfg=self.root/'config.json'
        ev.write_jsonl(q,[{'query_id':'q','question':'q','source_url':'https://example.invalid/a',
                          'candidate_evidence':[{'chunk_id':'canonical-a'}]}])
        materials=self.root/'materials';data.review_materials(q,materials,self.bundle)
        questions=materials/'questions.jsonl'
        ev.write_json(cfg,self.config);self.adapter.allow_model_calls=True
        with patch.object(self.adapter,'retrieve',return_value={'ranked_chunk_ids':['canonical-b'],'backend_latency_ms':1}):
            export_questions(questions,cfg,self.adapter,self.root/'runs')
        packet=rag_review.pool(self.bundle/'corpus.jsonl',questions,[self.root/'runs'/f'{m}.json' for m in ev.MODES],self.root/'packet.json')
        self.assertEqual({c['chunk_id'] for c in packet['questions'][0]['candidates']},{'canonical-a','canonical-b'})
        self.assertEqual(ev.read_jsonl(questions)[0]['label_status'],'pending_review')
        with self.assertRaises(ValueError): ev.validate(questions,self.bundle/'corpus.jsonl')

    def test_pending_export_keeps_failed_questions_without_scoring(self):
        q=self.root/'q.jsonl'; cfg=self.root/'config.json'
        ev.write_jsonl(q,[{'query_id':'q','question':'q','label_status':'pending_review','corpus_sha256':self.manifest['corpus_sha256']}])
        ev.write_json(cfg,self.config);self.adapter.allow_model_calls=True
        with patch.object(self.adapter,'retrieve',side_effect=RuntimeError('secret')):
            summary=export_questions(q,cfg,self.adapter,self.root/'runs')
        self.assertEqual(summary['failed_calls'],3);self.assertFalse((self.root/'runs/report.json').exists())
        for mode in ev.MODES:
            row=json.loads((self.root/'runs'/f'{mode}.json').read_text())['queries'][0]
            self.assertEqual(row['ranked_chunk_ids'],[]);self.assertEqual(row['error'],'RuntimeError')


if __name__ == '__main__': unittest.main()

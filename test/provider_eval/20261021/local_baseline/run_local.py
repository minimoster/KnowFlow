"""One finite CPU-local SciFact experiment. No networking or credential inputs."""
import sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='4'
os.environ['TOKENIZERS_PARALLELISM']='false'
sys.path.insert(0,str(ROOT/'deps'))
import csv,hashlib,json,math,platform,time,statistics
from collections import Counter,defaultdict
from datetime import datetime,timezone
import numpy as np
import pyarrow.parquet as pq
from core import BM25,rrf,rank,Limits,MODES
from local_models import Model
from upstream.rag_eval import metrics,p95

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(p,obj):
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def utc():return datetime.now(timezone.utc).isoformat()
def inputs():
    assets=json.loads((ROOT/'asset_manifest.json').read_text(encoding='utf-8'))
    for a in assets['files']:
        if sha(ROOT/a['path'])!=a['sha256']:raise ValueError('asset changed: '+a['path'])
    corpus=pq.read_table(ROOT/'data/scifact/corpus/corpus-00000-of-00001.parquet').to_pylist()
    queries=pq.read_table(ROOT/'data/scifact/queries/queries-00000-of-00001.parquet').to_pylist()
    ds={str(x['_id']):x for x in corpus};qs={str(x['_id']):x['text'] for x in queries}
    if len(ds)!=len(corpus) or len(qs)!=len(queries):raise ValueError('duplicate dataset IDs')
    qrels=defaultdict(set)
    with (ROOT/'data/scifact-qrels/test.tsv').open(encoding='utf-8') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            if int(row['score'])>0:qrels[row['query-id']].add(row['corpus-id'])
    if len(ds)!=5183 or len(qrels)!=300:raise ValueError('unexpected official split cardinality')
    if any(q not in qs or not gold<=ds.keys() for q,gold in qrels.items()):raise ValueError('qrels identity mismatch')
    docids=sorted(ds,key=lambda s:int(s));queryids=sorted(qrels,key=lambda s:int(s))
    texts=[(ds[d].get('title','')+' '+ds[d]['text']).strip() for d in docids]
    timingids=sorted(queryids,key=lambda q:hashlib.sha256(('knowflow-scifact-timing-v1:'+q).encode()).hexdigest())[:40]
    return assets,docids,texts,qs,qrels,queryids,timingids

def main():
    assets,docids,texts,queries,qrels,qids,timingids=inputs()
    out=ROOT/'run_01';out.mkdir(exist_ok=False)
    plan={'schema':'knowflow-public-local-eval-v1','dataset':'BEIR SciFact official test qrels','corpus_documents':5183,
      'unique_test_queries':300,'quality_retrievals':900,'timing_query_ids':timingids,'timing_repetitions':3,
      'timing_retrievals':360,'total_retrieval_attempt_cap':1260,'retry_cap':0,'local_inference_call_cap':2300,
      'run_wall_seconds_cap':2700,'threads_per_onnx_session':4,'concurrent_requests':1,
      'warmup':'Complete quality pass before timing repetitions; models and document vectors remain resident.',
      'query_embedding_cache':False,'result_cache':False,'document_embedding_cache':'generated once, hash-bound, resident for retrieval',
      'top_k':5,'candidate_k':20,'hybrid_per_branch_k':40,'rrf_k':60,
      'dense_max_wordpieces':256,'rerank_max_wordpieces':512,'dense_pooling':'masked mean then L2 normalize',
      'dense_search':'exact cosine on normalized 384-dim genuine transformer embeddings',
      'lexical_search':'local BM25 k1=1.2 b=0.75; lowercase ASCII alphanumeric tokenization, no stemming',
      'reranker':'local ONNX MS MARCO cross-encoder raw logits, full document up to 512 wordpieces',
      'quantization':'official quint8_avx2 ONNX exports','test_query_ids':qids,
      'model_pins':assets['models'],'dataset_pins':assets['datasets'],'asset_manifest_sha256':sha(ROOT/'asset_manifest.json'),
      'source_sha256':{p:sha(ROOT/p) for p in ['run_local.py','core.py','local_models.py','upstream/rag_eval.py']},
      'production_reuse':'Existing KnowFlow metrics/p95 functions copied unchanged. RRF formula/branch pool policy ported. Actual production Go runtime, Milvus, ES and chat reranker not executed.',
      'api_cost_usd':0,'local_energy_or_hardware_cost_usd':None,'external_model_requests_cap':0,
      'measurement_scope':'Public English document retrieval; not the Chinese historical corpus or deployed production performance.'}
    write_json(out/'plan.json',plan)
    events=(out/'trace.jsonl').open('x',encoding='utf-8');sequence=0
    def emit(e):
        nonlocal sequence
        sequence+=1;e={'seq':sequence,'time_utc':utc(),**e}
        events.write(json.dumps(e,ensure_ascii=False,allow_nan=False)+'\n');events.flush()
    limits=Limits(plan['run_wall_seconds_cap'],plan['local_inference_call_cap'])
    start=time.perf_counter();emit({'event':'run_started','plan_sha256':sha(out/'plan.json')})
    rows=[];error=None
    try:
        dense=Model(ROOT/'models/dense','dense',limits,emit)
        reranker=Model(ROOT/'models/rerank','rerank',limits,emit)
        # Contract probes are explicitly outside quality and timing denominators.
        v,_=dense.infer(['A scientific paper.','A different scientific paper.'],'contract_probe')
        if v.shape!=(2,384) or not np.allclose(np.linalg.norm(v,axis=1),1,atol=1e-5):raise ValueError('embedding contract')
        s,_=reranker.infer([('paper','A scientific paper.'),('paper','A kitchen recipe.')],'contract_probe')
        if s.shape!=(2,):raise ValueError('reranker contract')
        index_start=time.perf_counter();chunks=[]
        for offset in range(0,len(texts),32):
            vec,_=dense.infer(texts[offset:offset+32],'document_embedding');chunks.append(vec)
            if offset%512==0:print('document_embeddings',offset+len(vec),'/',len(texts),flush=True)
        matrix=np.concatenate(chunks);np.save(out/'document_embeddings.npy',matrix,allow_pickle=False)
        lexical=BM25(texts)
        write_json(out/'index_manifest.json',{'documents':len(docids),'dimensions':int(matrix.shape[1]),'document_ids':docids,
           'matrix_sha256':sha(out/'document_embeddings.npy'),'plan_sha256':sha(out/'plan.json'),'build_seconds':time.perf_counter()-index_start})
        del chunks
        def retrieve(question,mode):
            backend=time.perf_counter();components={};calls=[]
            t=time.perf_counter();q,u=dense.infer([question],'query_embedding');calls.append(u);components['query_embedding_ms']=(time.perf_counter()-t)*1000
            t=time.perf_counter();dense_ids=rank(matrix@q[0],20 if mode=='vector' else 40);components['dense_search_ms']=(time.perf_counter()-t)*1000
            if mode=='vector':chosen=dense_ids
            else:
                t=time.perf_counter();lex_ids=lexical.search(question,40);components['bm25_ms']=(time.perf_counter()-t)*1000
                t=time.perf_counter();chosen=rrf(dense_ids,lex_ids,60,20);components['rrf_ms']=(time.perf_counter()-t)*1000
            if mode=='hybrid_rerank' and len(chosen)>1:
                t=time.perf_counter();scores,u=reranker.infer([(question,texts[i]) for i in chosen],'rerank');calls.append(u)
                chosen=[chosen[i] for i in sorted(range(len(chosen)),key=lambda i:(-float(scores[i]),i))];components['rerank_ms']=(time.perf_counter()-t)*1000
            return [docids[i] for i in chosen[:5]],(time.perf_counter()-backend)*1000,components,calls
        tasks=[]
        for i,qid in enumerate(qids):
            for mode in MODES[i%3:]+MODES[:i%3]:tasks.append(('quality',0,qid,mode))
        for repeat in range(1,4):
            for i,qid in enumerate(timingids):
                shift=(i+repeat)%3
                for mode in MODES[shift:]+MODES[:shift]:tasks.append(('timing',repeat,qid,mode))
        for n,(phase,repeat,qid,mode) in enumerate(tasks,1):
            if time.monotonic()>=limits.deadline:raise TimeoutError('run deadline before next retrieval')
            outer=time.perf_counter();emit({'event':'retrieval_started','attempt_id':n,'phase':phase,'repeat':repeat,'query_id':qid,'mode':mode})
            row={'attempt_id':n,'phase':phase,'repeat':repeat,'query_id':qid,'mode':mode,'ranked_document_ids':[],
                 'status':'error','error':None,'backend_latency_ms':None,'usage':None,'api_cost_usd':0}
            try:
                ranked,ms,components,usage=retrieve(queries[qid],mode)
                if len(ranked)>5 or len(set(ranked))!=len(ranked) or not set(ranked)<=set(docids):raise ValueError('ranking identity contract')
                row.update(status='success',ranked_document_ids=ranked,backend_latency_ms=ms,components_ms=components,usage=usage)
            except Exception as exc:row['error']=type(exc).__name__+': '+str(exc)[:200]
            row['outer_latency_ms']=(time.perf_counter()-outer)*1000
            # Relevance is used only after retrieval returns; never passed to a model.
            row.update(metrics(row['ranked_document_ids'],qrels[qid]));row['hit_at_5']=int(bool(row['hit_ids']))
            rows.append(row);emit({'event':'retrieval_finished',**row})
            if n%60==0:print('retrievals',n,'/',len(tasks),'phase',phase,flush=True)
            if row['error']:raise RuntimeError('stop after retained retrieval failure')
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)[:350];emit({'event':'run_stopped','error':error})
    finally:
        events.close()
    # Recompute solely from the saved trace, including every started completed attempt.
    raw=[json.loads(s) for s in (out/'trace.jsonl').read_text(encoding='utf-8').splitlines()]
    saved=[x for x in raw if x['event']=='retrieval_finished']
    inf=[x for x in raw if x['event']=='inference_finished']
    report={'status':'complete' if len(saved)==1260 and error is None else 'partial','error':error,
      'unique_test_queries':300,'planned_quality_retrievals':900,'started_quality_retrievals':sum(x['event']=='retrieval_started' and x['phase']=='quality' for x in raw),
      'planned_timing_retrievals':360,'completed_timing_retrievals':sum(x['phase']=='timing' for x in saved),
      'actual_external_model_http':0,'api_cost_usd':0,'local_energy_or_hardware_cost_usd':None,
      'local_inference_calls_started':limits.calls,'local_inference_errors':sum(x['status']!='success' for x in inf),
      'wordpiece_usage_by_purpose':{},'run_seconds':time.perf_counter()-start,'modes':{},'plan_sha256':sha(out/'plan.json'),'trace_sha256':sha(out/'trace.jsonl')}
    for purpose in ['contract_probe','document_embedding','query_embedding','rerank']:
        relevant=[x for x in inf if x['purpose']==purpose]
        report['wordpiece_usage_by_purpose'][purpose]={'completed_calls':len(relevant),'input_wordpieces':sum((x.get('usage') or {}).get('input_wordpieces_including_special',0) for x in relevant) if all(x.get('usage') for x in relevant) else None,'usage_complete':all(x.get('usage') for x in relevant)}
    for mode in MODES:
        quality=[x for x in saved if x['phase']=='quality' and x['mode']==mode];timing=[x for x in saved if x['phase']=='timing' and x['mode']==mode]
        report['modes'][mode]={'quality_queries':len(quality),'hit_count':sum(x['hit_at_5'] for x in quality),
           **{m:statistics.mean(x[m] for x in quality) if quality else None for m in ['hit_at_5','recall_at_5','mrr_at_5']},
           'errors':sum(x['status']!='success' for x in quality),'timing_samples':len(timing),
           'warmed_backend_p95_ms':p95([x['backend_latency_ms'] for x in timing]) if timing and all(x['backend_latency_ms'] is not None for x in timing) else None,
           'warmed_outer_p95_ms':p95([x['outer_latency_ms'] for x in timing]) if timing else None,
           'warmed_backend_mean_ms':statistics.mean(x['backend_latency_ms'] for x in timing) if timing and all(x['backend_latency_ms'] is not None for x in timing) else None}
    write_json(out/'report.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return 0 if report['status']=='complete' else 2
if __name__=='__main__':raise SystemExit(main())

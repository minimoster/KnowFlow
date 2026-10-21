"""Read saved output and independently recompute metrics; no inference imports."""
import csv,hashlib,json,math,statistics
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent;RUN=ROOT/'run_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
plan=json.loads((RUN/'plan.json').read_text(encoding='utf-8'))
report=json.loads((RUN/'report.json').read_text(encoding='utf-8'))
events=[json.loads(x) for x in (RUN/'trace.jsonl').read_text(encoding='utf-8').splitlines()]
assert report['status']=='complete', 'partial execution must not become a complete result'
assert sha(RUN/'trace.jsonl')==report['trace_sha256']
assert sha(RUN/'plan.json')==report['plan_sha256']
for p,h in plan['source_sha256'].items():assert sha(ROOT/p)==h,p
gold=defaultdict(set)
with (ROOT/'data/scifact-qrels/test.tsv').open(encoding='utf-8') as f:
    for x in csv.DictReader(f,delimiter='\t'):
        if int(x['score'])>0:gold[x['query-id']].add(x['corpus-id'])
starts=[x for x in events if x['event']=='retrieval_started']
ends=[x for x in events if x['event']=='retrieval_finished']
assert len(starts)==len(ends)==1260
assert [x['attempt_id'] for x in starts]==list(range(1,1261))
assert [x['attempt_id'] for x in ends]==list(range(1,1261))
infs=[x for x in events if x['event']=='inference_started'];infe=[x for x in events if x['event']=='inference_finished']
assert len(infs)==len(infe)==report['local_inference_calls_started']<=plan['local_inference_call_cap']
assert [x['call_id'] for x in infs]==[x['call_id'] for x in infe]
assert all(x['usage']['external_api_calls']==0 and x['usage']['api_cost_usd']==0 for x in infe)
recomputed={};quality_ranks={};stable=0
for mode in ['vector','hybrid','hybrid_rerank']:
    quality=[x for x in ends if x['phase']=='quality' and x['mode']==mode]
    timing=[x for x in ends if x['phase']=='timing' and x['mode']==mode]
    assert len(quality)==300 and {x['query_id'] for x in quality}==set(plan['test_query_ids'])
    assert len(timing)==120 and Counter((x['query_id'],x['repeat']) for x in timing)==Counter((q,n) for q in plan['timing_query_ids'] for n in [1,2,3])
    vals=[]
    for x in quality:
        ranked=x['ranked_document_ids'];relevant=gold[x['query_id']]
        assert len(ranked)==len(set(ranked))<=5
        hits=set(ranked)&relevant
        rr=next((1/(i+1) for i,d in enumerate(ranked) if d in relevant),0)
        v=(int(bool(hits)),len(hits)/len(relevant),rr);vals.append(v)
        assert v==(x['hit_at_5'],x['recall_at_5'],x['mrr_at_5'])
        quality_ranks[(mode,x['query_id'])]=ranked
    stable+=sum(x['ranked_document_ids']==quality_ranks[(mode,x['query_id'])] for x in timing)
    quantile=lambda key:sorted(x[key] for x in timing)[math.ceil(.95*len(timing))-1]
    recomputed[mode]={'hit_count':sum(x[0] for x in vals),'queries':300,'hit_at_5':statistics.mean(x[0] for x in vals),
      'recall_at_5':statistics.mean(x[1] for x in vals),'mrr_at_5':statistics.mean(x[2] for x in vals),
      'warmed_backend_p95_ms':quantile('backend_latency_ms'),'warmed_outer_p95_ms':quantile('outer_latency_ms'),
      'warmed_timing_samples':120,'errors':sum(x['status']!='success' for x in quality)}
    for key in ['hit_count','hit_at_5','recall_at_5','mrr_at_5','warmed_backend_p95_ms','warmed_outer_p95_ms']:
        assert recomputed[mode][key]==report['modes'][mode][key]
result={'scope':'Real local CPU inference on public BEIR SciFact; benchmark adapter, not deployed Chinese KnowFlow stack',
 'unique_queries':300,'corpus_documents':5183,'positive_qrels':sum(map(len,gold.values())),
 'modes':recomputed,'quality_attempts':900,'timing_attempts':360,'attempt_errors':sum(x['status']!='success' for x in ends),
 'stable_repeated_rankings':stable,'total_repeated_rankings':360,'external_model_http':0,'provider_api_cost_usd':0,
 'hardware_and_energy_cost_usd':None,'local_inference_calls':len(infs),'run_seconds':report['run_seconds'],
 'start_utc':events[0]['time_utc'],'last_record_utc':events[-1]['time_utc'],
 'model_item_counts':dict(Counter({p:sum(x['items'] for x in infs if x['purpose']==p) for p in {x['purpose'] for x in infs}})),
 'source_and_trace_hashes_verified':True,'independent_metric_recomputation_passed':True,
 'limitations':['Local exact cosine replaces Milvus approximate L2; normalized vectors make cosine/L2 ranking mathematically equivalent only for this chosen representation.',
 'Local lexical tokenization/BM25 replaces Elasticsearch; the production analyzer/index was not run.',
 'Quantized local MS MARCO cross-encoder replaces production chat listwise reranker; 512-token pair input replaces 200-character previews.',
 'Warmed timing repeats 40 fixed queries three times per mode on one CPU session; it is not a concurrency/load benchmark or 120 independent questions.',
 'Public qrels are independently supplied; no new human review or private Chinese-corpus result is claimed.',
 'No dataset text, model weights, credentials or user private files are uploaded.'],
 'evidence':{p:sha(RUN/p) for p in ['plan.json','trace.jsonl','report.json','index_manifest.json']}}
write(ROOT/'final_metrics.json',result)
write(ROOT/'delivery_hashes.json',{p:sha(ROOT/p) for p in ['final_metrics.json','asset_manifest.json','environment.json','core.py','local_models.py','run_local.py','audit_results.py','offline_contract_tests.log']})
print(json.dumps(result,ensure_ascii=False))

"""Approved single-use provider worker. Preparation never invokes live mode."""
import sys,os,json,time,hashlib,subprocess,csv
from pathlib import Path
from datetime import datetime,timezone
sys.dont_write_bytecode=True
os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'knowflow_public_eval_20261021'
sys.path.insert(0,str(BASE/'deps'));sys.path.insert(0,str(BASE))
from contracts import *
from candidate_freeze import candidates_from_cached_vectors,freeze_one
from upstream.rag_eval import metrics,p95
import numpy as np
import pyarrow.parquet as pq

def utc():return datetime.now(timezone.utc).isoformat()
def save(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2)
def preflight(approval_path):
    approval=json.loads(Path(approval_path).read_text(encoding='utf-8'))
    plan=json.loads((ROOT/'phase_plan.pending.json').read_text(encoding='utf-8'))
    seal=json.loads((ROOT/'readiness_seal.json').read_text(encoding='utf-8'))
    if approval.get('status')!='approved' or approval.get('purpose')!='KnowFlow public SciFact Zhipu embedding and DeepSeek rerank':raise ValueError('new purpose-specific approval required')
    if approval.get('plan_sha256')!=digest((ROOT/'phase_plan.pending.json').read_bytes()):raise ValueError('approval plan identity')
    if approval.get('readiness_seal_sha256')!=digest((ROOT/'readiness_seal.json').read_bytes()):raise ValueError('approval source identity')
    if not approval.get('memory_only_zhipu_input') or not approval.get('read_existing_deepseek_target') or not approval.get('accept_budget_factor_not_exchange_rate'):raise ValueError('credential/currency approval incomplete')
    for name,h in seal['files'].items():
        if digest((ROOT/name).read_bytes())!=h:raise ValueError('frozen source/input mismatch')
    if (ROOT/'approved_run_01').exists():raise ValueError('existing run: reconcile, never restart')
    return plan
def go_bridge(mode,query,candidates,raw=''):
    p=subprocess.run([str(ROOT/'bridge.exe')],input=canonical({'mode':mode,'query':query,'candidates':candidates,'raw':raw})+b'\n',capture_output=True,timeout=10)
    if p.returncode:raise RuntimeError('production bridge failure')
    d=json.loads(p.stdout)
    if d.get('error'):raise ValueError('production rerank rejected output')
    return d

def live(approval_path):
    plan=preflight(approval_path) # before any credential stdin read
    secrets=json.load(sys.stdin)
    if set(secrets)!={'zhipu','deepseek'} or any(not isinstance(v,str) or not v for v in secrets.values()):raise ValueError('private stdin missing')
    # Local inputs are read/verified before activation. No keys written to artifacts.
    rows=[json.loads(x) for x in (ROOT/'frozen_inputs/embedding_inputs.jsonl').read_text(encoding='utf-8').splitlines()]
    batches=json.loads((ROOT/'frozen_inputs/embedding_batches.json').read_text(encoding='utf-8'))
    lookup={(x['kind'],x['id']):x for x in rows}
    docs=pq.read_table(BASE/'data/scifact/corpus/corpus-00000-of-00001.parquet').to_pylist()
    texts={str(x['_id']):(x.get('title','')+' '+x['text']).strip() for x in docs}
    docids=[x['id'] for x in rows if x['kind']=='corpus'];qids=plan['query_ids']
    queries={x['id']:x['text'] for x in rows if x['kind']=='query'}
    out=ROOT/'approved_run_01';out.mkdir(exist_ok=False)
    start=time.monotonic();deadline=start+plan['window_seconds']-plan['tail_seconds_no_dispatch']
    save(out/'activation.json',{'start_utc':utc(),'window_seconds':plan['window_seconds'],'dispatch_seconds':plan['window_seconds']-plan['tail_seconds_no_dispatch'],'plan_sha256':digest((ROOT/'phase_plan.pending.json').read_bytes()),'approved_plan_sha256':digest(Path(approval_path).read_bytes())})
    emb=Ledger(out/'embedding_ledger.jsonl',86,'8.421888',deadline);chat=Ledger(out/'deepseek_ledger.jsonl',300,'2',deadline)
    audit=(out/'trace.jsonl').open('x',encoding='utf-8');rankings={};status='stopped';reason=None;frozen=[]
    def emit(e):audit.write(json.dumps({'time_utc':utc(),**e},allow_nan=False)+'\n');audit.flush()
    try:
        vectors={}
        for b in batches:
            rid='embedding-'+str(b['batch_id']);items=[lookup[(b['kind'],i)]['text'] for i in b['ids']]
            payload=embedding_payload(items,2048)
            if digest(payload)!=b['request_sha256']:raise ValueError('embedding payload identity')
            reserve=Decimal(len(items)*3072)*Decimal('.5')/Decimal(1000000)
            t=time.perf_counter();data=budgeted_dispatch(emb,rid,ZHIPU_ENDPOINT,payload,secrets['zhipu'],reserve)
            (out/(rid+'.response.json')).write_bytes(data)
            body=strict_json(data);cost=embedding_cost(body.get('usage'),len(items))
            try:v,usage=validate_embeddings(data,len(items),2048)
            except Exception:emb.settle(rid,cost,'validation_error');raise
            emb.settle(rid,cost,'success');emit({'stage':'embedding','request_id':rid,'items':len(items),'usage':usage,'cost_CNY':None if cost is None else str(cost),'request_latency_ms':(time.perf_counter()-t)*1000})
            if emb.stopped:raise RuntimeError('unknown embedding cost')
            for i,vec in zip(b['ids'],v):vectors[(b['kind'],i)]=vec
        d=np.asarray([vectors[('corpus',i)] for i in docids],dtype=np.float32);q=np.asarray([vectors[('query',i)] for i in qids],dtype=np.float32)
        np.save(out/'document_vectors.npy',d,allow_pickle=False);np.save(out/'query_vectors.npy',q,allow_pickle=False)
        save(out/'vector_manifest.json',{'provider':'zhipu','model':'embedding-3','dimensions':2048,'complete':True,'input_manifest_sha256':plan['input_manifest_sha256'],'document_vectors_sha256':digest((out/'document_vectors.npy').read_bytes()),'query_vectors_sha256':digest((out/'query_vectors.npy').read_bytes()),'document_ids':docids,'query_ids':qids})
        d,q,bm=candidates_from_cached_vectors(d,q,[texts[i] for i in docids])
        # Freeze every candidate/prompt/payload before the FIRST DeepSeek request.
        for i,qid in enumerate(qids):
            c=freeze_one(d,q[i],bm,queries[qid],docids,[texts[i] for i in docids]);prompt=go_bridge('prompt',queries[qid],c['candidates'])
            payload=chat_payload(prompt['system'],prompt['user'])
            if len(prompt['system'].encode())+len(prompt['user'].encode())>5199:raise ValueError('visible prompt byte ceiling')
            frozen.append({'query_id':qid,**c,'payload':strict_json(payload),'payload_sha256':digest(payload),'template_sha256':prompt['template_sha256']})
        with (out/'frozen_candidates_payloads.jsonl').open('x',encoding='utf-8') as f:
            for row in frozen:f.write(canonical(row).decode()+'\n')
        emit({'stage':'candidate_freeze','queries':300,'sha256':digest((out/'frozen_candidates_payloads.jsonl').read_bytes())})
        for row in frozen:
            rid='rerank-'+row['query_id'];payload=canonical(row['payload']);t=time.perf_counter()
            try:data=budgeted_dispatch(chat,rid,DEEPSEEK_ENDPOINT,payload,secrets['deepseek'],DEEPSEEK_CALL_RESERVATION)
            except Exception:
                if rid in chat.calls:rankings[row['query_id']]={'status':'failed','ids':[]}
                raise
            rankings[row['query_id']]={'status':'failed','ids':[]}
            (out/(rid+'.response.json')).write_bytes(data)
            body=strict_json(data);cost=chat_cost(body.get('usage'))
            try:
                raw,usage,model,fingerprint=validate_chat_response(data)
                answer=go_bridge('parse',queries[row['query_id']],row['candidates'],raw)
            except Exception:
                rankings[row['query_id']]={'status':'failed','ids':[]};chat.settle(rid,cost,'validation_error');raise
            chat.settle(rid,cost,'success')
            if chat.stopped:
                rankings[row['query_id']]={'status':'failed_unknown_usage','ids':[]};raise RuntimeError('unknown cost')
            rankings[row['query_id']]={'status':'success','ids':[x['id'] for x in answer['results']],'request_latency_ms':(time.perf_counter()-t)*1000}
            emit({'stage':'rerank','query_id':row['query_id'],'usage':usage,'cost_USD':str(cost),'model':model,'system_fingerprint':fingerprint,'request_latency_ms':(time.perf_counter()-t)*1000,'ranked_ids':rankings[row['query_id']]['ids']})
        status='complete'
    except Exception as exc:reason=type(exc).__name__;emit({'stage':'stopped','error_type':reason})
    finally:
        secrets.clear();emb.close();chat.close();audit.close()
        # Truth enters scoring only after provider processing; never prompt builders.
        gold={}
        with (BASE/'data/scifact-qrels/test.tsv').open(encoding='utf-8') as f:
            for x in csv.DictReader(f,delimiter='\t'):
                if int(x['score'])>0:gold.setdefault(x['query-id'],set()).add(x['corpus-id'])
        results=[]
        for qid in qids:
            result=rankings.get(qid,{'status':'not_started','ids':[]});score=None
            if result['status']!='not_started':score=metrics(result['ids'],gold[qid]);score['hit_at_5']=int(bool(score['hit_ids']))
            results.append({'query_id':qid,**result,'metrics':score})
        comparison={}
        def summarize(scores):
            return {'queries':len(scores),**{k:sum(x[k] for x in scores)/len(scores) if scores else None for k in ['hit_at_5','recall_at_5','mrr_at_5']}}
        if len(frozen)==300:
            for mode,field in [('vector','vector_top5'),('hybrid','hybrid_top5')]:
                s=[]
                for row in frozen:
                    m=metrics(row[field],gold[row['query_id']]);m['hit_at_5']=int(bool(m['hit_ids']));s.append(m)
                comparison[mode]=summarize(s)
        comparison['hybrid_deepseek_rerank']=summarize([r['metrics'] for r in results if r['metrics'] is not None])
        save(out/'results.json',{'status':status,'reason':reason,'planned_queries':300,'started_queries':len(rankings),'unstarted_queries':300-len(rankings),'results':results,'comparison':comparison,'end_utc':utc(),'elapsed_seconds':time.monotonic()-start,'actual_bill_unknown':True,'embedding_HTTP_attempts':len(emb.calls),'deepseek_HTTP_attempts':len(chat.calls),'conservative_USD_settled_and_held':str(sum(x['held'] for x in chat.calls.values())),'CNY_settled_and_held':str(sum(x['held'] for x in emb.calls.values())),'latency_scope':'rerank provider request + output validation; not warmed P95 or full retrieval latency'})
    return 0 if status=='complete' else 2

if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1] not in {'--check','--approved-plan'}:raise SystemExit('requires explicit approval plan; no implicit activation')
    if sys.argv[1]=='--check':preflight(sys.argv[2]);print('preflight passed; no credential read or activation')
    else:raise SystemExit(live(sys.argv[2]))

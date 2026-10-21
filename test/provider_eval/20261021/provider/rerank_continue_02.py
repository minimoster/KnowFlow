"""One explicitly approved continuation; frozen inputs and original deadline only."""
import sys,json,time,csv,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
from decimal import Decimal
sys.dont_write_bytecode=True
from contracts import (digest,canonical,strict_json,Ledger,budgeted_dispatch,
    DEEPSEEK_ENDPOINT,DEEPSEEK_CALL_RESERVATION,validate_chat_response,chat_cost)
ROOT=Path(__file__).resolve().parent
FIRST=ROOT/'approved_run_01'
OLD=ROOT/'rerank_continuation_01'
OUT=ROOT/'rerank_continuation_02'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def save(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,allow_nan=False,indent=2)
def utc():return datetime.now(timezone.utc)

def prior_account(events):
    reserved={};settled={}
    for e in events:
        if e['event']=='reserved':
            if e['request_id'] in reserved:raise ValueError('duplicate reservation')
            reserved[e['request_id']]=e
        if e['event']=='settled':
            if e['request_id'] not in reserved or e['request_id'] in settled:raise ValueError('invalid settlement')
            settled[e['request_id']]=e
    known=Decimal(0);held=Decimal(0);unknown=[]
    for rid,r in reserved.items():
        e=settled.get(rid)
        if e is None or e['cost'] is None:
            held+=Decimal(r['amount']);unknown.append(rid)
        else:
            c=Decimal(e['cost'])
            if not c.is_finite() or not 0<=c<=Decimal(r['amount']):raise ValueError('invalid prior cost')
            known+=c
    return {'attempts':len(reserved),'known':known,'held':held,'unknown':unknown}

def bounds(account,activation,total_http_cap,embedding_attempts,now):
    deadline=datetime.fromisoformat(activation['start_utc'])+timedelta(seconds=activation['dispatch_seconds'])
    remaining=(deadline-now).total_seconds()
    if remaining<=0:raise ValueError('original dispatch deadline expired')
    calls=total_http_cap-embedding_attempts-account['attempts']
    budget=Decimal('2')-account['known']-account['held']
    if calls<=0 or budget<DEEPSEEK_CALL_RESERVATION:raise ValueError('cumulative count/budget insufficient')
    return calls,budget,remaining,deadline

def preflight(approval_path):
    a=read(Path(approval_path))
    if a.get('status')!='approved' or a.get('purpose')!='same-phase one-time rerank continuation':raise ValueError('explicit continuation approval required')
    if a.get('total_http_cap')!=388 or a.get('max_new_attempts')!=65 or not a.get('read_existing_deepseek_target'):raise ValueError('exact continuation scope required')
    if OUT.exists():raise ValueError('continuation already exists; never restart')
    for name,h in a['source_hashes'].items():
        if digest((ROOT/name).read_bytes())!=h:raise ValueError('continuation source mismatch')
    for name,h in a['previous_hashes'].items():
        if digest((ROOT/name).read_bytes())!=h:raise ValueError('prior evidence changed')
    for name,h in read(ROOT/'readiness_seal.json')['files'].items():
        if digest((ROOT/name).read_bytes())!=h:raise ValueError('original source/input mismatch')
    result=read(OLD/'results.json');activation=read(FIRST/'activation.json')
    if result['status']!='stopped' or result['reason']!='ValueError':raise ValueError('unexpected prior stop')
    failure=read(OLD/'validation_failure_1137.json')
    if failure.get('gate_reason')!='incomplete scores' or failure.get('score_rows')!=19:raise ValueError('prior failed output identity')
    account=prior_account(rows(FIRST/'deepseek_ledger.jsonl')+rows(OLD/'deepseek_ledger.jsonl'))
    emb=prior_account(rows(FIRST/'embedding_ledger.jsonl'))
    if account['attempts']!=237 or account['unknown']!=['rerank-693'] or emb['attempts']!=86 or emb['unknown']:raise ValueError('unexpected cumulative state')
    if account['known']!=Decimal('0.1445544') or account['held']!=DEEPSEEK_CALL_RESERVATION:raise ValueError('prior risk identity')
    calls,budget,seconds,deadline=bounds(account,activation,a['total_http_cap'],emb['attempts'],utc())
    todo=[x['query_id'] for x in result['results'] if x['status']!='success']
    if len(todo)!=65 or todo[0]!='1137' or calls!=65:raise ValueError('unexpected remaining queries')
    frozen=rows(FIRST/'frozen_candidates_payloads.jsonl')
    if len(frozen)!=300 or len({x['query_id'] for x in frozen})!=300:raise ValueError('candidate completeness')
    for x in frozen:
        if digest(canonical(x['payload']))!=x['payload_sha256']:raise ValueError('frozen payload changed')
    return a,result,activation,account,budget,deadline,todo,frozen

def bridge(query,candidates,raw):
    p=subprocess.run([str(ROOT/'bridge.exe')],input=canonical({'mode':'parse','query':query,'candidates':candidates,'raw':raw})+b'\n',capture_output=True,timeout=10)
    if p.returncode:raise ValueError('production bridge exit')
    answer=strict_json(p.stdout)
    if answer.get('error'):raise ValueError('production bridge rejected')
    return [x['id'] for x in answer['results']]

def live(approval_path):
    a,old,activation,prior,budget,deadline,todo,frozen=preflight(approval_path)
    secret=json.load(sys.stdin)
    if set(secret)!={'deepseek'} or not isinstance(secret['deepseek'],str) or not secret['deepseek']:raise ValueError('private stdin required')
    remaining=(deadline-utc()).total_seconds()
    if remaining<=0:secret.clear();raise ValueError('original deadline expired before continuation')
    OUT.mkdir(exist_ok=False)
    save(OUT/'continuation.json',{'continued_utc':utc().isoformat(),'original_activation':activation,'dispatch_deadline_utc':deadline.isoformat(),'new_window':False,'prior_attempts':237,'prior_known_USD':str(prior['known']),'prior_unknown_risk_USD':str(prior['held']),'new_attempt_cap':65,'new_USD_available':str(budget),'total_HTTP_cap':388,'approval_sha256':digest(Path(approval_path).read_bytes()),'previous_hashes':a['previous_hashes']})
    ledger=Ledger(OUT/'deepseek_ledger.jsonl',65,budget,time.monotonic()+remaining)
    trace=(OUT/'trace.jsonl').open('x',encoding='utf-8')
    def emit(d):trace.write(json.dumps({'time_utc':utc().isoformat(),**d},allow_nan=False)+'\n');trace.flush()
    queries={x['id']:x['text'] for x in rows(ROOT/'frozen_inputs/embedding_inputs.jsonl') if x['kind']=='query'}
    byid={x['query_id']:x for x in frozen};updates={};status='stopped';reason=None
    try:
        for qid in todo:
            x=byid[qid];rid='rerank-'+qid+'-continuation2';t=time.perf_counter()
            try:data=budgeted_dispatch(ledger,rid,DEEPSEEK_ENDPOINT,canonical(x['payload']),secret['deepseek'],DEEPSEEK_CALL_RESERVATION)
            except Exception:
                if rid in ledger.calls:updates[qid]={'status':'failed','ids':[],'attempt_id':rid}
                raise
            updates[qid]={'status':'failed','ids':[],'attempt_id':rid}
            (OUT/(rid+'.response.json')).write_bytes(data)
            cost=None
            try:
                body=strict_json(data);cost=chat_cost(body.get('usage'))
                raw,usage,model,fingerprint=validate_chat_response(data)
                ids=bridge(queries[qid],x['candidates'],raw)
            except Exception:ledger.settle(rid,cost,'validation_error');raise
            ledger.settle(rid,cost,'success')
            if ledger.stopped:raise ValueError('unknown usage; no further dispatch')
            latency=(time.perf_counter()-t)*1000
            updates[qid]={'status':'success','ids':ids,'attempt_id':rid,'request_latency_ms':latency}
            emit({'stage':'rerank','query_id':qid,'attempt_id':rid,'usage':usage,'cost_USD':str(cost),'model':model,'system_fingerprint':fingerprint,'request_latency_ms':latency,'ranked_ids':ids})
        status='complete'
    except Exception as e:reason=type(e).__name__;emit({'stage':'stopped','error_type':reason})
    finally:
        secret.clear();ledger.close();trace.close()
        gold={}
        with (ROOT.parent/'knowflow_public_eval_20261021/data/scifact-qrels/test.tsv').open(encoding='utf-8') as f:
            for x in csv.DictReader(f,delimiter='\t'):
                if int(x['score'])>0:gold.setdefault(x['query-id'],set()).add(x['corpus-id'])
        merged=[]
        for previous in old['results']:
            qid=previous['query_id'];r={'query_id':qid,**updates.get(qid,previous)}
            if qid in updates:r['prior_status']=previous['status']
            ids=r['ids'];truth=gold[qid]
            r['metrics']=None if r['status']=='not_started' else {'hit_at_5':int(bool(set(ids)&truth)),'recall_at_5':len(set(ids)&truth)/len(truth),'mrr_at_5':next((1/i for i,x in enumerate(ids,1) if x in truth),0)}
            merged.append(r)
        scores=[x['metrics'] for x in merged if x['metrics'] is not None]
        summary={k:sum(x[k] for x in scores)/len(scores) for k in ['hit_at_5','recall_at_5','mrr_at_5']}
        summary['queries']=len(scores)
        save(OUT/'results.json',{'status':status,'reason':reason,'planned_queries':300,'results':merged,'comparison':{**old['comparison'],'hybrid_deepseek_rerank':summary},'new_HTTP_attempts':len(ledger.calls),'cumulative_HTTP_attempts':86+237+len(ledger.calls),'cumulative_DeepSeek_USD_settled_and_held':str(prior['known']+prior['held']+sum(x['held'] for x in ledger.calls.values())),'end_utc':utc().isoformat(),'original_start_utc':activation['start_utc'],'original_failure_preserved':True,'actual_bill_unknown':True})
    return 0 if status=='complete' else 2

if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1] not in {'--check','--approved-plan'}:raise SystemExit('explicit continuation approval required')
    if sys.argv[1]=='--check':
        preflight(sys.argv[2]);print('continuation preflight passed; no credentials, requests, or activation')
    else:raise SystemExit(live(sys.argv[2]))

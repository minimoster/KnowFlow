"""One separately approved post-benchmark request; sealed scores are never edited."""
import json,sys,time,os
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
sys.dont_write_bytecode=True
from contracts import (Ledger,one_http,canonical,digest,strict_json,strict_scores,
    validate_chat_response,chat_cost,chat_payload,DEEPSEEK_ENDPOINT,DEEPSEEK_CALL_RESERVATION)
from rerank_continue_03 import prior_account,bridge
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'diagnostic_1137_01'
RUNS=['approved_run_01','rerank_continuation_01','rerank_continuation_02','rerank_continuation_03']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def utc():return datetime.now(timezone.utc).isoformat()
def save(p,d):
    with p.open('x',encoding='utf-8') as f:
        json.dump(d,f,ensure_ascii=False,allow_nan=False,indent=2);f.flush();os.fsync(f.fileno())

def preflight(approval_path):
    a=read(Path(approval_path))
    if a.get('status')!='approved' or a.get('purpose')!='one post-benchmark diagnostic for query 1137':raise ValueError('diagnostic-specific approval required')
    if a.get('query_id')!='1137' or a.get('max_requests')!=1 or a.get('timeout_seconds')!=30 or not a.get('read_existing_deepseek_target'):raise ValueError('exact one-call scope required')
    if OUT.exists():raise ValueError('diagnostic already exists; never restart')
    for name,h in a['hashes'].items():
        if digest((ROOT/name).read_bytes())!=h:raise ValueError('sealed evidence/source changed')
    events=[e for n in RUNS for e in rows(ROOT/n/'deepseek_ledger.jsonl')]
    account=prior_account(events)
    if account['attempts']!=302 or account['known']!=Decimal('0.1844208') or account['held']!=Decimal('0.3158016') or account['unknown']!=['rerank-693']:raise ValueError('prior budget identity mismatch')
    available=Decimal('2')-account['known']-account['held']
    if available<DEEPSEEK_CALL_RESERVATION:raise ValueError('budget insufficient before dispatch')
    sealed=read(ROOT/'rerank_continuation_03/results.json')
    if sealed['status']!='complete':raise ValueError('benchmark not complete')
    target=next(x for x in sealed['results'] if x['query_id']=='1137')
    if target['status']!='failed' or target['ids']:raise ValueError('sealed failure identity changed')
    frozen=next(x for x in rows(ROOT/'approved_run_01/frozen_candidates_payloads.jsonl') if x['query_id']=='1137')
    payload=canonical(frozen['payload'])
    if digest(payload)!=a['payload_sha256'] or digest(payload)!=frozen['payload_sha256']:raise ValueError('frozen payload changed')
    d=frozen['payload']
    if payload!=chat_payload(d['messages'][0]['content'],d['messages'][1]['content']):raise ValueError('model parameters changed')
    if len(frozen['candidates'])!=20:raise ValueError('candidate count changed')
    query=next(x['text'] for x in rows(ROOT/'frozen_inputs/embedding_inputs.jsonl') if x['kind']=='query' and x['id']=='1137')
    return a,account,available,frozen,payload,query

def coverage(raw):
    s=raw.strip()
    for prefix in ['```json','```']:
        if s.startswith(prefix):s=s[len(prefix):]
    if s.endswith('```'):s=s[:-3]
    d=strict_json(s.strip());values=d.get('scores')
    if not isinstance(values,list):return {'rows':None,'missing':list(range(20)),'valid_complete':False}
    indices=[x.get('idx') for x in values if isinstance(x,dict)]
    ints=[i for i in indices if type(i) is int]
    result={'rows':len(values),'indices':indices,'missing':sorted(set(range(20))-set(ints)),
            'duplicate_indices':sorted({i for i in ints if ints.count(i)>1}),
            'unexpected_indices':[i for i in indices if type(i) is not int or not 0<=i<20],
            'valid_complete':False}
    try:strict_scores(raw);result['valid_complete']=True
    except Exception as e:result['validation_condition']=str(e)
    return result

def live(approval_path):
    a,prior,available,frozen,payload,query=preflight(approval_path)
    private=json.load(sys.stdin)
    if set(private)!={'deepseek'} or not isinstance(private['deepseek'],str) or not private['deepseek']:raise ValueError('private stdin required')
    OUT.mkdir(exist_ok=False)
    ledger=Ledger(OUT/'deepseek_ledger.jsonl',1,available,time.monotonic()+30)
    rid='diagnostic-1137-01';cost=None;usage=None;verdict='unknown';condition=None;cov=None;ids=[];model=None;fp=None
    ledger.reserve(rid,DEEPSEEK_CALL_RESERVATION,digest(payload))
    save(OUT/'dispatch_receipt.json',{'request_id':rid,'pre_transport_utc':utc(),'timeout_seconds':30,'endpoint':DEEPSEEK_ENDPOINT,'payload_sha256':digest(payload),'reservation_USD':str(DEEPSEEK_CALL_RESERVATION),'prior_known_USD':str(prior['known']),'prior_unknown_held_USD':str(prior['held']),'cumulative_DeepSeek_cap_USD':'2','benchmark_window_reopened':False,'diagnostic_request_limit':1})
    started=time.perf_counter()
    try:
        try:data=one_http(DEEPSEEK_ENDPOINT,payload,private['deepseek'],ledger.deadline-time.monotonic())
        except Exception as e:
            ledger.settle(rid,None,'transport_error');condition=type(e).__name__;raise
        (OUT/'response.json').write_bytes(data)
        try:
            body=strict_json(data);usage=body.get('usage');cost=chat_cost(usage)
            model=body.get('model');fp=body.get('system_fingerprint')
            raw=body['choices'][0]['message']['content'];cov=coverage(raw)
            raw,usage,model,fp=validate_chat_response(data)
            ids=bridge(query,frozen['candidates'],raw)
            if not cov['valid_complete']:raise ValueError('independent completeness gate failed')
        except Exception as e:
            ledger.settle(rid,cost,'validation_error');verdict='invalid';condition=str(e);raise
        ledger.settle(rid,cost,'success')
        if ledger.stopped:verdict='unknown';condition='unknown usage';ids=[]
        else:verdict='valid'
    except Exception as e:
        if condition is None:condition=type(e).__name__
    finally:
        private.clear();ledger.close()
        held=sum(x['held'] for x in ledger.calls.values())
        save(OUT/'result.json',{'request_id':rid,'query_id':'1137','diagnostic_only':True,'verdict':verdict,'validation_condition':condition,'score_coverage':cov,'ranked_ids':ids,'usage':usage,'model':model,'system_fingerprint':fp,'known_cost_USD':None if cost is None else str(cost),'new_settled_or_held_USD':str(held),'prior_known_USD':str(prior['known']),'prior_unknown_held_USD':str(prior['held']),'cumulative_DeepSeek_settled_and_held_USD':str(prior['known']+prior['held']+held),'new_dispatch_attempts':1,'cumulative_DeepSeek_attempts':303,'cumulative_all_provider_attempts':389,'elapsed_ms':(time.perf_counter()-started)*1000,'end_utc':utc(),'actual_bill_known':False,'sealed_benchmark_unchanged':True})
    return 0 if verdict=='valid' else 2

if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1] not in {'--check','--approved-plan'}:raise SystemExit('explicit one-call diagnostic approval required')
    if sys.argv[1]=='--check':preflight(sys.argv[2]);print('diagnostic preflight passed; no credential read or request')
    else:raise SystemExit(live(sys.argv[2]))

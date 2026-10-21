"""Offline-testable provider contracts. No credential lookup or activation CLI."""
import hashlib,json,math,time,threading
from decimal import Decimal
from pathlib import Path

DEEPSEEK_ENDPOINT='https://api.deepseek.com/chat/completions'
ZHIPU_ENDPOINT='https://open.bigmodel.cn/api/paas/v4/embeddings'
DEEPSEEK_MODEL='deepseek-flash'
DEEPSEEK_VERSION='DeepSeek-V4.1-Flash-1017'
MAX_OUTPUT=1024
MAX_BODY=32768
CONTEXT_INPUT_RESERVATION=1048576
DEEPSEEK_CALL_RESERVATION=Decimal('0.3158016')

def digest(data):return hashlib.sha256(data).hexdigest()
def canonical(obj):return json.dumps(obj,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')
def unique_object(items):
    d={}
    for k,v in items:
        if k in d:raise ValueError('duplicate JSON key')
        d[k]=v
    return d
def strict_json(raw):
    return json.loads(raw,object_pairs_hook=unique_object,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def strict_scores(raw,n=20):
    # Additional transport schema gate; production parser remains unchanged.
    s=raw.strip()
    for prefix in ['```json','```']:
        if s.startswith(prefix):s=s[len(prefix):]
    if s.endswith('```'):s=s[:-3]
    d=strict_json(s.strip())
    if not isinstance(d,dict) or set(d)!={'scores'} or not isinstance(d['scores'],list) or len(d['scores'])!=n:raise ValueError('incomplete scores')
    seen=set()
    for x in d['scores']:
        if not isinstance(x,dict) or set(x)!={'idx','score'}:raise ValueError('invalid score fields')
        if type(x['idx']) is not int or not 0<=x['idx']<n or x['idx'] in seen:raise ValueError('invalid index')
        if type(x['score']) not in (int,float) or not math.isfinite(x['score']) or not 0<=x['score']<=10 or int(x['score'])!=x['score']:raise ValueError('invalid score')
        seen.add(x['idx'])
    return d
def chat_payload(system,user):
    d={'model':DEEPSEEK_MODEL,'messages':[{'role':'system','content':system},{'role':'user','content':user}],
       'thinking':{'type':'disabled'},'temperature':0,'max_tokens':MAX_OUTPUT,'stream':False}
    b=canonical(d)
    if len(b)>MAX_BODY:raise ValueError('request body cap')
    return b
def validate_chat_response(body):
    d=strict_json(body)
    if d.get('model') not in {DEEPSEEK_MODEL,DEEPSEEK_VERSION}:raise ValueError('unexpected model')
    if not isinstance(d.get('system_fingerprint'),str) or not d['system_fingerprint']:raise ValueError('missing fingerprint')
    choices=d.get('choices')
    if not isinstance(choices,list) or len(choices)!=1:raise ValueError('invalid choices')
    c=choices[0];m=c.get('message',{})
    if c.get('finish_reason')!='stop' or m.get('tool_calls') or m.get('reasoning_content'):raise ValueError('incomplete or unexpected generation')
    raw=m.get('content')
    if not isinstance(raw,str):raise ValueError('missing content')
    strict_scores(raw)
    return raw,d.get('usage'),d['model'],d['system_fingerprint']
def chat_cost(usage):
    if not isinstance(usage,dict):return None
    names=['prompt_tokens','completion_tokens','total_tokens']
    if any(type(usage.get(k)) is not int or usage[k]<0 for k in names):return None
    p,o,total=(usage[k] for k in names)
    if total!=p+o or p>CONTEXT_INPUT_RESERVATION or o>MAX_OUTPUT:return None
    if 'prompt_cache_hit_tokens' in usage or 'prompt_cache_miss_tokens' in usage:
        h=usage.get('prompt_cache_hit_tokens');m=usage.get('prompt_cache_miss_tokens')
        if type(h) is not int or type(m) is not int or h<0 or m<0 or h+m!=p:return None
    # Conservative settlement: all prompt tokens at peak cache-miss price.
    return (Decimal(p)*Decimal('0.30')+Decimal(o)*Decimal('1.20'))/Decimal(1000000)
def embedding_cost(usage,items):
    if not isinstance(usage,dict):return None
    n=usage.get('total_tokens')
    if type(n) is not int or not 0<=n<=items*3072:return None
    return Decimal(n)*Decimal('0.5')/Decimal(1000000)
def embedding_payload(items,dimensions=2048):
    if dimensions not in {256,512,1024,2048} or not 0<len(items)<=64:raise ValueError('embedding request bounds')
    if any(not isinstance(x,str) or not x or len(x.encode('utf-8'))>3072 for x in items):raise ValueError('frozen input byte policy')
    return canonical({'model':'embedding-3','input':items,'dimensions':dimensions})
def validate_embeddings(body,count,dimensions):
    d=strict_json(body)
    if d.get('model')!='embedding-3':raise ValueError('embedding model mismatch')
    data=d.get('data')
    if not isinstance(data,list) or len(data)!=count:raise ValueError('embedding count')
    ordered=[None]*count
    for row in data:
        i=row.get('index');v=row.get('embedding')
        if type(i) is not int or not 0<=i<count or ordered[i] is not None:raise ValueError('embedding index')
        if not isinstance(v,list) or len(v)!=dimensions or any(type(x) not in (int,float) or not math.isfinite(x) for x in v):raise ValueError('embedding dimension/nonfinite')
        if not any(x!=0 for x in v):raise ValueError('zero embedding')
        ordered[i]=v
    return ordered,d.get('usage')

class Ledger:
    """Reserve before transport; unknown stays reserved and stops new dispatch.

    Single-owner exclusive file. Existing ledger is never resumed/reset by this
    preparation module. Costs use one caller-declared currency, no implicit FX.
    """
    def __init__(self,path,cap,budget,deadline):
        self.path=Path(path);self.file=self.path.open('x',encoding='utf-8')
        self.cap=cap;self.budget=Decimal(str(budget));self.deadline=deadline
        self.calls={};self.stopped=False;self.lock=threading.Lock()
        self._write({'event':'init','cap':cap,'budget':str(budget),'deadline':deadline})
    def _write(self,event):
        self.file.write(json.dumps(event,allow_nan=False)+'\n');self.file.flush()
        import os;os.fsync(self.file.fileno())
    def reserve(self,request_id,reservation,payload_sha):
        with self.lock:
            amount=Decimal(str(reservation)) if reservation is not None else None
            if amount is None or not amount.is_finite() or amount<=0:raise ValueError('unknown reservation')
            if self.stopped or time.monotonic()>=self.deadline:raise RuntimeError('stopped/deadline')
            if request_id in self.calls or len(self.calls)>=self.cap:raise RuntimeError('duplicate/call cap')
            if any(not x['settled'] for x in self.calls.values()):raise RuntimeError('unsettled prior transport')
            if sum(x['held'] for x in self.calls.values())+amount>self.budget:raise RuntimeError('budget cap')
            self._write({'event':'reserved','request_id':request_id,'amount':str(amount),'payload_sha256':payload_sha})
            self.calls[request_id]={'held':amount,'settled':False};return request_id
    def settle(self,request_id,cost,outcome):
        with self.lock:
            c=self.calls[request_id]
            if c['settled']:raise RuntimeError('duplicate settlement')
            amount=Decimal(str(cost)) if cost is not None else None
            if amount is None or not amount.is_finite() or amount<0 or amount>c['held']:
                self.stopped=True;amount=None
            if outcome!='success':self.stopped=True
            self._write({'event':'settled','request_id':request_id,'cost':None if amount is None else str(amount),'outcome':outcome,'unknown_risk_retained':amount is None})
            if amount is not None:c['held']=amount
            c['settled']=True
    def close(self):self.file.close()

def one_http(endpoint,payload,key,remaining_seconds,connection_factory=None):
    """One actual stdlib transport attempt; no SDK/redirect/retry/proxy/env keys.
    Not reachable through any activation CLI in this preparation delivery.
    """
    if endpoint not in {DEEPSEEK_ENDPOINT,ZHIPU_ENDPOINT}:raise ValueError('endpoint allowlist')
    if remaining_seconds<=0:raise TimeoutError('dispatch deadline')
    from urllib.parse import urlsplit
    import http.client
    u=urlsplit(endpoint);factory=connection_factory or http.client.HTTPSConnection
    con=factory(u.hostname,timeout=min(60,remaining_seconds))
    try:
        con.request('POST',u.path,body=payload,headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
        limit=8*1024*1024 if endpoint==ZHIPU_ENDPOINT else 256*1024
        response=con.getresponse();data=response.read(limit+1)
        if len(data)>limit:raise ValueError('response size cap')
        if response.status!=200:raise RuntimeError('provider HTTP status '+str(response.status))
        return data
    finally:con.close()

def budgeted_dispatch(ledger,request_id,endpoint,payload,key,reservation,connection_factory=None):
    ledger.reserve(request_id,reservation,digest(payload))
    try:
        data=one_http(endpoint,payload,key,ledger.deadline-time.monotonic(),connection_factory)
    except Exception:
        ledger.settle(request_id,None,'transport_error');raise
    # Settlement is deliberately the caller's responsibility after validation;
    # until then, the reservation is held. Live activation requires all gates.
    return data

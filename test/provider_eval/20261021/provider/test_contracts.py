import json,time,unittest,threading,subprocess,uuid
from pathlib import Path
from contracts import *

class FakeConnection:
    status=200;body=b'{}';raised=False;sent=0;closed=0
    def __init__(self,host,timeout):self.host=host
    def request(self,*args,**kwargs):
        type(self).sent+=1
        if self.raised:raise TimeoutError('synthetic timeout')
    def getresponse(self):return self
    def read(self,n):return self.body[:n]
    def close(self):type(self).closed+=1

def scores():return json.dumps({'scores':[{'idx':i,'score':5} for i in range(20)]})
def response():return {'model':DEEPSEEK_VERSION,'system_fingerprint':'offline-test-fingerprint','choices':[{'finish_reason':'stop','message':{'content':scores()}}],'usage':{'prompt_tokens':100,'completion_tokens':200,'total_tokens':300,'prompt_cache_hit_tokens':20,'prompt_cache_miss_tokens':80}}

class Contracts(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).parent/'offline_test_records'/str(uuid.uuid4())
        self.root.mkdir(parents=True,exist_ok=False)
        FakeConnection.status=200;FakeConnection.body=b'{}';FakeConnection.raised=False;FakeConnection.sent=0;FakeConnection.closed=0
    def tearDown(self):pass # retain isolated test receipts; no filesystem cleanup
    def ledger(self,cap=2,budget='1',seconds=30):return Ledger(self.root/'ledger.jsonl',cap,budget,time.monotonic()+seconds)
    def test_controls(self):
        p=strict_json(chat_payload('system','user'));self.assertEqual(p['max_tokens'],1024);self.assertEqual(p['thinking'],{'type':'disabled'});self.assertFalse(p['stream'])
        with self.assertRaises(ValueError):chat_payload('s','x'*32768)
    def test_schema_and_production_success(self):
        d=response();raw,usage,_,_=validate_chat_response(canonical(d));self.assertIsNotNone(chat_cost(usage))
        candidates=[{'id':str(i),'score':1/(i+61),'source':'hybrid','chunk':{'content':'synthetic'}} for i in range(20)]
        p=subprocess.run([str(Path(__file__).parent/'bridge.exe')],input=canonical({'mode':'parse','query':'synthetic','candidates':candidates,'raw':raw})+b'\n',capture_output=True,timeout=10)
        out=json.loads(p.stdout);self.assertFalse(out.get('error'));self.assertEqual([x['id'] for x in out['results']],['0','1','2','3','4'])
    def test_schema_rejects_missing_null_duplicate_and_nonfinite(self):
        bad=[scores().replace('"score": 5','"score": null',1),scores().replace('"score": 5','"score": NaN',1),scores().replace('"score": 5','"score": true',1),scores().replace('"idx": 1','"idx": 0',1),'{}','{"scores":[],"scores":[]}']
        for raw in bad:
            with self.subTest(raw=raw[:40]),self.assertRaises(ValueError):strict_scores(raw)
    def test_length_and_judge_failure_never_pass(self):
        for change in ['length','content_filter','aborted','tool_calls']:
            d=response();d['choices'][0]['finish_reason']=change
            with self.assertRaises(ValueError):validate_chat_response(canonical(d))
        d=response();d['model']='wrong'
        with self.assertRaises(ValueError):validate_chat_response(canonical(d))
        d=response();d['system_fingerprint']=''
        with self.assertRaises(ValueError):validate_chat_response(canonical(d))
    def test_unknown_usage(self):
        self.assertIsNone(chat_cost(None));self.assertIsNone(chat_cost({'prompt_tokens':0}));d=response()['usage'];d['completion_tokens']=1025;self.assertIsNone(chat_cost(d))
    def test_embedding_contract(self):
        p=strict_json(embedding_payload(['a','b'],2048));self.assertEqual(p['dimensions'],2048)
        body={'model':'embedding-3','data':[{'index':1,'embedding':[1,2]},{'index':0,'embedding':[3,4]}],'usage':{'total_tokens':2}}
        v,_=validate_embeddings(canonical(body),2,2);self.assertEqual(v,[[3,4],[1,2]])
        body['data'][1]['index']=1
        with self.assertRaises(ValueError):validate_embeddings(canonical(body),2,2)
        with self.assertRaises(ValueError):embedding_payload(['a']*65)
        with self.assertRaises(ValueError):embedding_payload(['x'*3073])
    def test_reservation_precedes_transport(self):
        l=self.ledger(cap=1,budget='.05')
        try:
            with self.assertRaises(RuntimeError):budgeted_dispatch(l,'q1',DEEPSEEK_ENDPOINT,b'{}','DUMMY',.1,FakeConnection)
            with self.assertRaises(ValueError):budgeted_dispatch(l,'q1',DEEPSEEK_ENDPOINT,b'{}','DUMMY',None,FakeConnection)
            self.assertEqual(FakeConnection.sent,0)
        finally:l.close()
    def test_dispatch_and_no_retry_or_redirect(self):
        l=self.ledger(cap=1)
        try:
            FakeConnection.status=302
            with self.assertRaises(RuntimeError):budgeted_dispatch(l,'q1',DEEPSEEK_ENDPOINT,b'{}','DUMMY',.1,FakeConnection)
            self.assertEqual(FakeConnection.sent,1);self.assertTrue(l.stopped);self.assertEqual(l.calls['q1']['held'],Decimal('.1'))
            with self.assertRaises(RuntimeError):budgeted_dispatch(l,'q2',DEEPSEEK_ENDPOINT,b'{}','DUMMY',.1,FakeConnection)
            self.assertEqual(FakeConnection.sent,1)
        finally:l.close()
    def test_timeout_unknown_duplicate_terminal_and_restart(self):
        l=self.ledger()
        try:
            FakeConnection.raised=True
            with self.assertRaises(TimeoutError):budgeted_dispatch(l,'q1',DEEPSEEK_ENDPOINT,b'{}','DUMMY',.1,FakeConnection)
            with self.assertRaises(RuntimeError):l.settle('q1',0,'success')
            with self.assertRaises(FileExistsError):Ledger(self.root/'ledger.jsonl',2,1,time.monotonic()+30)
        finally:l.close()
    def test_expired_and_budget_gate_no_transport(self):
        l=self.ledger(seconds=-1)
        try:
            with self.assertRaises(RuntimeError):budgeted_dispatch(l,'q1',DEEPSEEK_ENDPOINT,b'{}','DUMMY',.1,FakeConnection)
            self.assertEqual(FakeConnection.sent,0)
        finally:l.close()
    def test_inflight_and_unknown_prevent_new_call(self):
        l=self.ledger()
        try:
            l.reserve('a',.5,'hash')
            with self.assertRaises(RuntimeError):l.reserve('b',.1,'hash')
            l.settle('a',None,'success')
            with self.assertRaises(RuntimeError):l.reserve('b',.1,'hash')
        finally:l.close()
    def test_parallel_reservations_do_not_oversubscribe(self):
        l=self.ledger(cap=1);passed=[]
        def go(i):
            try:l.reserve(str(i),.5,'hash');passed.append(i)
            except RuntimeError:pass
        ts=[threading.Thread(target=go,args=(i,)) for i in range(5)]
        try:
            for t in ts:t.start()
            for t in ts:t.join()
            self.assertEqual(len(passed),1)
        finally:l.close()
    def test_rolling_reservations_release_only_measured_surplus(self):
        l=self.ledger(cap=3,budget='.8')
        try:
            for i in range(3):
                l.reserve(str(i),'.5','hash');l.settle(str(i),'.1','success')
            self.assertEqual(sum(x['held'] for x in l.calls.values()),Decimal('.3'))
            with self.assertRaises(RuntimeError):l.reserve('fourth','.1','hash')
        finally:l.close()
if __name__=='__main__':unittest.main(verbosity=2)

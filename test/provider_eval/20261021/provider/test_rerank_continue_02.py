import unittest,io,json
from datetime import datetime,timezone,timedelta
from decimal import Decimal
from unittest.mock import patch
import rerank_continue_02 as r

class ContinuationTests(unittest.TestCase):
    def test_prior_unknown_is_carried(self):
        e=[{'event':'reserved','request_id':'a','amount':'0.3158016'}, {'event':'settled','request_id':'a','cost':None}]
        a=r.prior_account(e)
        self.assertEqual(a['held'],Decimal('0.3158016'));self.assertEqual(a['unknown'],['a'])
    def test_duplicate_settlement_rejected(self):
        e=[{'event':'reserved','request_id':'a','amount':'1'},{'event':'settled','request_id':'a','cost':'0.1'}]
        with self.assertRaises(ValueError):r.prior_account(e+[e[-1]])
    def test_cumulative_caps_and_same_deadline(self):
        t=datetime(2026,10,21,4,44,41,tzinfo=timezone.utc)
        a={'attempts':237,'known':Decimal('.1445544'),'held':Decimal('.3158016')}
        c,b,s,d=r.bounds(a,{'start_utc':t.isoformat(),'dispatch_seconds':3540},388,86,t+timedelta(seconds=1000))
        self.assertEqual((c,b,s),(65,Decimal('1.5396440'),2540));self.assertEqual(d,t+timedelta(seconds=3540))
        with self.assertRaises(ValueError):r.bounds(a,{'start_utc':t.isoformat(),'dispatch_seconds':3540},388,86,t+timedelta(seconds=3540))
    def test_pending_rejects_before_private_input(self):
        with patch.object(r,'read',return_value={'status':'pending'}),patch.object(r.sys,'stdin') as private:
            with self.assertRaises(ValueError):r.live('dummy')
            private.read.assert_not_called()
    def test_real_production_parser(self):
        c=[{'id':str(i),'chunk':{'content':'fixture'},'score':1/(61+i),'source':'hybrid'} for i in range(20)]
        raw=json.dumps({'scores':[{'idx':i,'score':i%10} for i in range(20)]})
        self.assertEqual(r.bridge('synthetic',c,raw),['9','19','8','18','7'])
    def test_transport_failure_has_one_attempt_and_unknown_risk(self):
        class Ledger:
            deadline=10**12
            def __init__(self):self.events=[]
            def reserve(self,*a):self.events.append(('reserve',a))
            def settle(self,*a):self.events.append(('settle',a))
        l=Ledger()
        with patch('contracts.one_http',side_effect=EOFError) as call:
            with self.assertRaises(EOFError):r.budgeted_dispatch(l,'a',r.DEEPSEEK_ENDPOINT,b'{}','DUMMY',r.DEEPSEEK_CALL_RESERVATION)
            self.assertEqual(call.call_count,1)
        self.assertEqual(l.events[-1],('settle',('a',None,'transport_error')))

if __name__=='__main__':unittest.main(verbosity=2)

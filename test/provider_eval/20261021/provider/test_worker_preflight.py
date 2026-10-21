import unittest,json
from pathlib import Path
from unittest.mock import patch
import worker

class Preflight(unittest.TestCase):
    def test_pending_never_reads_secret_or_activates(self):
        root=Path(__file__).parent
        with patch.object(worker.sys,'stdin') as secret:
            with self.assertRaises(ValueError):worker.live(root/'phase_plan.pending.json')
            secret.read.assert_not_called()
        self.assertFalse((root/'approved_run_01').exists())
    def test_production_parse_in_actual_worker_path(self):
        candidates=[{'id':str(i),'chunk':{'content':'fixture'},'score':1/(61+i),'source':'hybrid'} for i in range(20)]
        prompt=worker.go_bridge('prompt','synthetic',candidates)
        self.assertIn('[19]',prompt['user'])
        raw=json.dumps({'scores':[{'idx':i,'score':i%10} for i in range(20)]})
        result=worker.go_bridge('parse','synthetic',candidates,raw)
        self.assertEqual([x['id'] for x in result['results']],['9','19','8','18','7'])
if __name__=='__main__':unittest.main(verbosity=2)

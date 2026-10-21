import unittest
import numpy as np
from core import BM25,rrf,rank,Limits,score_rows
from upstream.rag_eval import metrics,p95

class ContractTests(unittest.TestCase):
    def test_real_metric_partial_and_rank(self):
        x=metrics(['b','a'],{'a','c'});self.assertEqual(x['recall_at_5'],.5);self.assertEqual(x['mrr_at_5'],.5)
    def test_failure_stays_zero_in_denominator(self):
        rows=[metrics(['a'],{'a'}),metrics([],{'a'})]
        self.assertEqual(score_rows(rows,'recall_at_5'),.5)
    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):metrics(['a','a'],{'a'})
    def test_missing_labels_rejected(self):
        with self.assertRaises(ValueError):metrics(['a'],set())
    def test_p95_nearest_rank(self):self.assertEqual(p95(list(range(1,121))),114)
    def test_rrf_equal_weight_and_tie(self):self.assertEqual(rrf([2,1],[1,2]),[1,2])
    def test_rrf_duplicate_rejected(self):
        with self.assertRaises(ValueError):rrf([1,1],[])
    def test_lexical_match_only(self):
        b=BM25(['alpha alpha','beta gamma','gamma']);self.assertEqual(b.search('alpha',3),[0]);self.assertEqual(b.search('unknown',3),[])
    def test_nonfinite_scores_rejected(self):
        with self.assertRaises(ValueError):rank([1,np.nan],2)
    def test_cap_does_not_reset_after_failure(self):
        x=Limits(max_inference_calls=1);x.reserve()
        with self.assertRaises(RuntimeError):x.reserve()
        self.assertEqual(x.calls,1)
    def test_expired_before_dispatch(self):
        x=Limits(seconds=-1)
        with self.assertRaises(TimeoutError):x.reserve()
        self.assertEqual(x.calls,0)
if __name__=='__main__':unittest.main(verbosity=2)

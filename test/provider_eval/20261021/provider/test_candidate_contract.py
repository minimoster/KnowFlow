import unittest
from candidate_freeze import *

class CandidateContract(unittest.TestCase):
    def test_same_space_and_provenance_required(self):
        with self.assertRaises(ValueError):candidates_from_cached_vectors(np.ones((20,384)),np.ones((1,2048)),['doc']*20)
        with self.assertRaises(ValueError):require_vector_provenance({'model':'all-MiniLM-L6-v2'},'x')
        with self.assertRaises(ValueError):candidates_from_cached_vectors(np.zeros((20,2048)),np.ones((1,2048)),['doc']*20)
    def test_actual_fixed_rrf_pool_and_scores(self):
        docs=np.zeros((20,2048),dtype=np.float32);docs[:,0]=1;query=docs[:1].copy()
        text=['science document '+str(i) for i in range(20)]
        matrix,q,bm=candidates_from_cached_vectors(docs,query,text)
        out=freeze_one(matrix,q[0],bm,'science',list(map(str,range(20))),text)
        self.assertEqual(len(out['candidates']),20)
        self.assertEqual(out['hybrid_top5'],[x['id'] for x in out['candidates'][:5]])
        self.assertEqual(out['vector_top5'],list(map(str,range(5))))
        self.assertAlmostEqual(out['candidates'][0]['score'],2/61)
if __name__=='__main__':unittest.main(verbosity=2)

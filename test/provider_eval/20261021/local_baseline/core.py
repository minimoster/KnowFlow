"""Local benchmark substitutes for storage/search; not Milvus or Elasticsearch."""
import math,re,time
from collections import Counter,defaultdict
import numpy as np

MODES=('vector','hybrid','hybrid_rerank')
def tokens(text):return re.findall(r'[a-z0-9]+',text.lower())
def rank(scores,k):
    scores=np.asarray(scores)
    if scores.ndim!=1 or not np.isfinite(scores).all():raise ValueError('invalid ranking scores')
    return np.lexsort((np.arange(len(scores)),-scores))[:k].tolist()
def rrf(a,b,k=60,limit=20):
    scores=defaultdict(float)
    for seq in [a,b]:
        if len(set(seq))!=len(seq):raise ValueError('duplicate search identity')
        for pos,doc in enumerate(seq,1):scores[doc]+=1/(k+pos)
    return sorted(scores,key=lambda doc:(-scores[doc],doc))[:limit]
class BM25:
    def __init__(self,texts,k1=1.2,b=.75):
        self.n=len(texts); self.k1=k1;self.b=b
        self.lengths=np.array([len(tokens(t)) for t in texts],dtype=np.float64)
        self.avg=float(self.lengths.mean()); self.index=defaultdict(list)
        if self.n==0 or self.avg<=0:raise ValueError('empty lexical corpus')
        for i,t in enumerate(texts):
            for term,tf in Counter(tokens(t)).items():self.index[term].append((i,tf))
    def search(self,text,k):
        scores=np.zeros(self.n,dtype=np.float64)
        for term in set(tokens(text)):
            posting=self.index.get(term,[])
            if not posting:continue
            ii=np.array([x[0] for x in posting]);tf=np.array([x[1] for x in posting])
            idf=math.log(1+(self.n-len(posting)+.5)/(len(posting)+.5))
            scores[ii]+=idf*tf*(self.k1+1)/(tf+self.k1*(1-self.b+self.b*self.lengths[ii]/self.avg))
        # ES match does not return zero-match documents.
        return [i for i in rank(scores,k) if scores[i]>0]
class Limits:
    def __init__(self,seconds=2700,max_inference_calls=2300):
        self.deadline=time.monotonic()+seconds;self.max_calls=max_inference_calls;self.calls=0
    def reserve(self):
        if time.monotonic()>=self.deadline:raise TimeoutError('fixed local run deadline')
        if self.calls>=self.max_calls:raise RuntimeError('local inference call cap')
        self.calls+=1
def score_rows(rows,metric):
    if not rows:raise ValueError('no started retrievals')
    return sum(x[metric] for x in rows)/len(rows)

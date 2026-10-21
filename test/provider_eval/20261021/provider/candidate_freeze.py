"""Future cached Zhipu-vector adapter. No inference, transport or qrels input."""
import sys,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'knowflow_public_eval_20261021'
sys.path.insert(0,str(BASE/'deps'));sys.path.insert(0,str(BASE))
import numpy as np
from core import BM25,rank,rrf

def candidates_from_cached_vectors(doc_vectors,query_vectors,texts):
    d=np.asarray(doc_vectors,dtype=np.float32);q=np.asarray(query_vectors,dtype=np.float32)
    if d.ndim!=2 or q.ndim!=2 or d.shape[1]!=q.shape[1] or len(d)!=len(texts) or len(d)<20:raise ValueError('vector identity/dimension')
    if d.shape[1]!=2048:raise ValueError('planned Zhipu dimensions; MiniLM mixing prohibited')
    if not np.isfinite(d).all() or not np.isfinite(q).all():raise ValueError('nonfinite vector')
    dn=np.linalg.norm(d,axis=1,keepdims=True);qn=np.linalg.norm(q,axis=1,keepdims=True)
    if (dn<=0).any() or (qn<=0).any():raise ValueError('zero vector')
    d=d/dn;q=q/qn;bm=BM25(texts)
    return d,q,bm

def freeze_one(matrix,query,bm,text,docids,fulltexts):
    a=rank(matrix@query,40);b=bm.search(text,40);chosen=rrf(a,b,60,20)
    scores={i:0.0 for i in chosen}
    for seq in [a,b]:
        for pos,i in enumerate(seq,1):
            if i in scores:scores[i]+=1/(60+pos)
    return {'vector_top5':[docids[i] for i in a[:5]],'hybrid_top5':[docids[i] for i in chosen[:5]],
            'candidates':[{'id':docids[i],'score':scores[i],'source':'hybrid','chunk':{'content':fulltexts[i]}} for i in chosen]}

def require_vector_provenance(meta,frozen_input_sha):
    required={'model':'embedding-3','dimensions':2048,'input_manifest_sha256':frozen_input_sha,'provider':'zhipu'}
    if any(meta.get(k)!=v for k,v in required.items()):raise ValueError('cached vector provenance mismatch')
    if not meta.get('complete') or not meta.get('document_vectors_sha256') or not meta.get('query_vectors_sha256'):raise ValueError('incomplete cached vector evidence')

def load_cached_vectors(meta,document_path,query_path,frozen_input_sha):
    require_vector_provenance(meta,frozen_input_sha)
    for name,path in [('document',document_path),('query',query_path)]:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=meta[name+'_vectors_sha256']:raise ValueError('vector file hash mismatch')
    d=np.load(document_path,allow_pickle=False);q=np.load(query_path,allow_pickle=False)
    if d.shape!=(5183,2048) or q.shape!=(300,2048):raise ValueError('complete corpus/query matrix required')
    return d,q

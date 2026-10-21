"""Freeze public embedding inputs; never imports/executes a neural model."""
import sys,os,json,hashlib,math,subprocess
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'knowflow_public_eval_20261021'
sys.path.insert(0,str(BASE/'deps'))
from contracts import canonical,digest,embedding_payload,chat_payload
import pyarrow.parquet as pq
from tokenizers import Tokenizer

def sha(p):return digest(p.read_bytes())
def prefix(text,cap=3072):return text.encode('utf-8')[:cap].decode('utf-8',errors='ignore')
def write_new(p,obj):
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def bridge(query,candidates):
    p=subprocess.run([str(ROOT/'bridge.exe')],input=canonical({'mode':'prompt','query':query,'candidates':candidates})+b'\n',capture_output=True,timeout=10)
    if p.returncode:raise RuntimeError('offline production bridge failed')
    d=json.loads(p.stdout)
    if d.get('error'):raise ValueError(d['error'])
    return d
def reference_text(system,user):
    # Text-only two-message, no tools/format/schema, chat mode; manually reviewed
    # upstream encoding.py lines 29, 35-36, 618-657, 731-740, 783.
    return '<｜begin▁of▁sentence｜><｜System｜>'+system+'<｜User｜>'+user+'<｜Assistant｜></think>'

def main():
    out=ROOT/'frozen_inputs';out.mkdir(exist_ok=False)
    assets=json.loads((BASE/'asset_manifest.json').read_text(encoding='utf-8'))
    for a in assets['files']:
        if a['path'].startswith('data/') and sha(BASE/a['path'])!=a['sha256']:raise ValueError('baseline data changed')
    docs=pq.read_table(BASE/'data/scifact/corpus/corpus-00000-of-00001.parquet').to_pylist()
    queries=pq.read_table(BASE/'data/scifact/queries/queries-00000-of-00001.parquet').to_pylist()
    plan=json.loads((BASE/'run_01/plan.json').read_text(encoding='utf-8'))
    qmap={str(x['_id']):x['text'] for x in queries};dmap={str(x['_id']):(x.get('title','')+' '+x['text']).strip() for x in docs}
    inputs=[]
    for kind,ids,lookup in [('corpus',sorted(dmap,key=int),dmap),('query',plan['test_query_ids'],qmap)]:
        for key in ids:
            original=lookup[key];text=prefix(original)
            inputs.append({'kind':kind,'id':key,'text':text,'source_text_sha256':digest(original.encode('utf-8')),'input_sha256':digest(text.encode('utf-8')),'source_bytes':len(original.encode('utf-8')),'input_bytes':len(text.encode('utf-8')),'truncated':text!=original})
    if len(inputs)!=5483:raise ValueError('input count')
    with (out/'embedding_inputs.jsonl').open('x',encoding='utf-8') as f:
        for row in inputs:f.write(canonical(row).decode('utf-8')+'\n')
    batches=[]
    for kind in ['corpus','query']:
        group=[x for x in inputs if x['kind']==kind]
        for start in range(0,len(group),64):
            batch=group[start:start+64];payload=embedding_payload([x['text'] for x in batch],2048)
            batches.append({'batch_id':len(batches)+1,'kind':kind,'ids':[x['id'] for x in batch],'items':len(batch),'request_sha256':digest(payload),'input_token_ceiling_from_endpoint_limit':len(batch)*3072})
    write_new(out/'embedding_batches.json',batches)
    # Bounds use the largest visible previews, NOT inferred original candidates.
    previews={k:(v[:200]+'…' if len(v)>200 else v) for k,v in dmap.items()}
    biggest=sorted(previews,key=lambda k:len(previews[k].encode('utf-8')),reverse=True)[:20]
    candidates=[{'id':k,'chunk':{'content':dmap[k]},'score':1/(i+61),'source':'hybrid'} for i,k in enumerate(biggest)]
    longest=max(plan['test_query_ids'],key=lambda k:len(qmap[k].encode('utf-8')))
    prompt=bridge(qmap[longest],candidates)
    write_new(out/'worst_visible_prompt_bound.json',{'purpose':'upper bound fixture only; not query candidates or an experiment','system_sha256':prompt['template_sha256'],'user_sha256':digest(prompt['user'].encode()),'max_query_bytes':len(qmap[longest].encode()),'sum_20_largest_preview_bytes':sum(len(previews[k].encode()) for k in biggest),'visible_message_bytes_ceiling':len(prompt['system'].encode())+len(prompt['user'].encode()),'body_bytes_example':len(chat_payload(prompt['system'],prompt['user']))})
    tokenizer=Tokenizer.from_file(str(ROOT/'provider_reference/tokenizer.json'))
    tokens=len(tokenizer.encode(reference_text(prompt['system'],prompt['user']),add_special_tokens=False).ids)
    meta={'status':'offline_inputs_frozen_not_approved','embedding_model':'embedding-3','dimensions':2048,'dimension_basis':'explicit default choice; no safe-file evidence of a fixed project requirement','corpus_items':5183,'query_items':300,'requests_no_retries':len(batches),'retry_cap':0,'input_policy':'first <=3072 UTF-8 bytes, valid codepoint boundary, title-space-text; all queries retained','truncated_documents':sum(x['truncated'] for x in inputs if x['kind']=='corpus'),'truncated_queries':sum(x['truncated'] for x in inputs if x['kind']=='query'),'input_token_ceiling_from_endpoint_limit':5483*3072,'embedding_CNY_reservation':'8.421888','tokenizer_count_worst_visible_example':tokens,'reference_bound_is_API_billing_guarantee':False,'hidden_API_overhead_bound':None,'input_manifest_sha256':sha(out/'embedding_inputs.jsonl'),'batch_manifest_sha256':sha(out/'embedding_batches.json'),'template_sha256':prompt['template_sha256'],'baseline_plan_sha256':sha(BASE/'run_01/plan.json'),'baseline_trace_sha256':sha(BASE/'run_01/trace.jsonl'),'old_top20_recovery':'blocked: query vectors and branch scores were not persisted','public_corpus_qrels_sent_to_providers':False}
    write_new(out/'manifest.json',meta)
    print(json.dumps(meta))
if __name__=='__main__':main()

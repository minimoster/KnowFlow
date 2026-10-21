import json,time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

class Model:
    def __init__(self,root,kind,limits,emit):
        ort.disable_telemetry_events()
        self.kind=kind;self.limits=limits;self.emit=emit
        self.tokenizer=Tokenizer.from_file(str(root/'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=256 if kind=='dense' else 512,strategy='longest_first')
        self.tokenizer.enable_padding(pad_id=0,pad_token='[PAD]')
        opt=ort.SessionOptions();opt.intra_op_num_threads=4;opt.inter_op_num_threads=1
        opt.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        self.session=ort.InferenceSession(str(root/'onnx/model_quint8_avx2.onnx'),sess_options=opt,providers=['CPUExecutionProvider'])
        self.input_names={x.name for x in self.session.get_inputs()}
        if not self.input_names <= {'input_ids','attention_mask','token_type_ids'}:raise ValueError('unexpected ONNX inputs')
    def infer(self,texts,purpose):
        if not 0<len(texts)<=32:raise ValueError('inference batch cap')
        self.limits.reserve();call=self.limits.calls;start=time.perf_counter()
        self.emit({'event':'inference_started','call_id':call,'model_kind':self.kind,'purpose':purpose,'items':len(texts)})
        try:
            encoded=self.tokenizer.encode_batch(texts)
            arrays={'input_ids':np.array([e.ids for e in encoded],dtype=np.int64),
                    'attention_mask':np.array([e.attention_mask for e in encoded],dtype=np.int64),
                    'token_type_ids':np.array([e.type_ids for e in encoded],dtype=np.int64)}
            output=self.session.run(None,{k:v for k,v in arrays.items() if k in self.input_names})[0]
            if self.kind=='dense':
                if output.ndim!=3 or output.shape[-1]!=384:raise ValueError('dense output shape')
                mask=arrays['attention_mask'][...,None]
                output=(output*mask).sum(axis=1)/np.maximum(mask.sum(axis=1),1)
                norms=np.linalg.norm(output,axis=1,keepdims=True)
                if np.any(norms<=0):raise ValueError('zero dense embedding')
                output=(output/norms).astype(np.float32)
            else:output=output.reshape(-1)
            if not np.isfinite(output).all():raise ValueError('nonfinite model output')
            usage={'input_wordpieces_including_special':int(arrays['attention_mask'].sum()),'padded_positions':int(arrays['input_ids'].size),'output_tokens':None,'external_api_calls':0,'api_cost_usd':0}
            self.emit({'event':'inference_finished','call_id':call,'model_kind':self.kind,'purpose':purpose,'status':'success','elapsed_ms':(time.perf_counter()-start)*1000,'usage':usage})
            return output,usage
        except Exception as exc:
            self.emit({'event':'inference_finished','call_id':call,'model_kind':self.kind,'purpose':purpose,'status':'error','error_type':type(exc).__name__,'elapsed_ms':(time.perf_counter()-start)*1000,'usage':None})
            raise

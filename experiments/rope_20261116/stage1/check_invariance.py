import json,torch,math
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset
p=Path(__file__).parent;meta=json.load(open(p/'independent_model.json'));torch.set_num_threads(2)
m=AutoModelForCausalLM.from_pretrained(meta['model'],revision=meta['revision'],attn_implementation='eager').eval();tok=AutoTokenizer.from_pretrained(meta['model'],revision=meta['revision']);data=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='test');x=tok('\n\n'.join(data['text']),return_tensors='pt').input_ids[:,:512];original={k:v.clone() for k,v in m.state_dict().items()};out={}
for dtype in [torch.float32,torch.float64]:
 m.to(dtype);m.load_state_dict(original)
 a=torch.full((8,),3*math.pi/16,dtype=dtype);c,s=a.cos(),a.sin();u=torch.eye(64,dtype=dtype);u[:16,:16]=torch.cat([torch.cat([torch.diag(c),-torch.diag(s)],1),torch.cat([torch.diag(s),torch.diag(c)],1)],0)
 with torch.inference_mode():
  ref=m(x,labels=x,use_cache=False);rlog=ref.logits.clone();rloss=float(ref.loss)
  for layer in m.gpt_neox.layers:
   proj=layer.attention.query_key_value;w=proj.weight.reshape(8,3,64,512);b=proj.bias.reshape(8,3,64)
   for h in range(8):
    for z in [0,1]:w[h,z].copy_(u@w[h,z]);b[h,z].copy_(u@b[h,z])
  new=m(x,labels=x,use_cache=False);dif=new.logits-rlog
 out[str(dtype)]={'max_abs':float(dif.abs().max()),'rms':float(dif.square().mean().sqrt()),'reference_max_abs':float(rlog.abs().max()),'reference_loss':rloss,'rotated_loss':float(new.loss),'loss_difference':float(new.loss)-rloss}
 print(dtype,out[str(dtype)],flush=True)
json.dump(out,open(p/'invariance_diagnostic.json','w'),indent=2)

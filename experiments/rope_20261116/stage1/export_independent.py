"""Serialize real 4/8-bit QK patches, scales and rotated FP32 biases; verify decoding.
Not a packed integer inference kernel. V and all other weights stay unchanged.
"""
from pathlib import Path
import json,math,hashlib,numpy as np,torch
from transformers import AutoModelForCausalLM
p=Path(__file__).parent;torch.set_num_threads(1);r=json.load(open(p/'independent_results.json'));meta=r['model'];m=AutoModelForCausalLM.from_pretrained(meta['model'],revision=meta['revision']).eval()
# Match torch float32 trigonometry in the experiment exactly.
def rot(angle):
 a=torch.full((8,),angle);c,s=torch.cos(a),torch.sin(a);u=torch.eye(64);u[:16,:16]=torch.cat([torch.cat([torch.diag(c),-torch.diag(s)],1),torch.cat([torch.diag(s),torch.diag(c)],1)],0);return u
bases=[rot(a) for a in [0,math.pi/16,math.pi/8,3*math.pi/16]];manifest={}
items=dict(r['results'])
if (p/'budget_results.json').exists():items.update(json.load(open(p/'budget_results.json')))
for mode,item in items.items():
 path=item['allocation']
 if path is None:continue
 payload=bytearray(bytes(j for j,a in path));checks=0
 for i,(j,a) in enumerate(path):
  li,h=divmod(i,8);proj=m.gpt_neox.layers[li].attention.query_key_value;w=proj.weight.detach().reshape(8,3,64,512);bias=proj.bias.detach().reshape(8,3,64)
  for z,b in enumerate([(4,4),(4,8),(8,4),(8,8)][j]):
   wr=bases[a]@w[h,z];sc=wr.abs().amax(1)/(2**(b-1)-1);codes=(wr/sc.clamp_min(1e-30)[:,None]).round().clamp(-(2**(b-1)-1),2**(b-1)-1).to(torch.int8).numpy()
   if b==4:
    c=(codes.astype(np.int16)+8).astype(np.uint8);raw=(c[:,::2]|(c[:,1::2]<<4)).tobytes();pk=np.frombuffer(raw,np.uint8).reshape(64,-1);back=np.stack([pk&15,pk>>4],-1).reshape(64,512).astype(np.int16)-8
   else:raw=codes.tobytes();back=np.frombuffer(raw,np.int8).reshape(64,512)
   scales=sc.numpy().astype('<f4').tobytes();reconstructed=torch.from_numpy(back.astype(np.float32))*torch.from_numpy(np.frombuffer(scales,'<f4').copy())[:,None];assert torch.equal(reconstructed,torch.from_numpy(codes.copy()).float()*sc[:,None])
   br=bases[a]@bias[h,z];biasraw=br.numpy().astype('<f4').tobytes();assert np.array_equal(np.frombuffer(biasraw,'<f4'),br.numpy());payload.extend(raw);payload.extend(scales);payload.extend(biasraw);checks+=1
 file=p/(mode+'_pythia_qk.bin');file.write_bytes(payload);expected=48*(128*512*4//8+128*4)+sum([0,1,1,2][j] for j,a in path)*64*512*4//8+48+48*128*4;assert len(payload)==expected
 manifest[mode]={'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'projection_groups_roundtripped':checks,'biases_included':True,'expected_value_equality':True}
json.dump(manifest,open(p/'independent_packed_manifest.json','w'),indent=2);print(json.dumps(manifest,indent=2))

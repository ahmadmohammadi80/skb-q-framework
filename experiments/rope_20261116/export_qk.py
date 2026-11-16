"""Export real Q/K payloads; verify round-trip values and byte-identical budgets.
Format: 90 uint8 option codes, then per group Q[192,576] payload+192 FP32 scales,
K[64,576] payload+64 FP32 scales. 4-bit codes use q+8, low nibble first.
This is a Q/K patch, not a complete deployable integer-kernel model.
"""
from pathlib import Path
exec(compile(Path(__file__).with_name('run_full.py').read_text().split('data=load_dataset')[0],__file__,'exec'))
torch.set_num_threads(1)
import struct
pub=json.load(open(out/'published_rule_results.json'))
manifest={}
for mode in ['identity','fixed','conditioned','published_rule_transplant']:
 if mode=='published_rule_transplant':
  path=pub['allocation'];model.load_state_dict(original,strict=False)
 else:
  score,path=allocate(mode);apply(path)
 payload=bytearray(bytes(j for j,b in path));checks=0
 for idx,(choice,bi) in enumerate(path):
  li,g=divmod(idx,3);att=model.model.layers[li].self_attn;qb,kb=[(4,4),(4,8),(8,4),(8,8)][choice]
  # Quantizer scale is taken from the original rotated weights, not the dequantized weight.
  for proj,rows,sl,bits in [('q',192,slice(g*192,(g+1)*192),qb),('k',64,slice(g*64,(g+1)*64),kb)]:
   key=f'model.layers.{li}.self_attn.{proj}_proj.weight';w=original[key][sl]
   basis=rotation(torch.tensor(pub['angles'][li],dtype=torch.float32)) if mode=='published_rule_transplant' else bases[bi]
   w=torch.cat([basis@z for z in w.split(64)],0);limit=2**(bits-1)-1;sc=w.abs().amax(1)/limit
   codes=(w/sc.clamp_min(1e-30)[:,None]).round().clamp(-limit,limit).to(torch.int8).numpy()
   if bits==4:
    z=(codes.astype(np.int16)+8).astype(np.uint8);raw=(z[:,::2]|(z[:,1::2]<<4)).tobytes()
    packed=np.frombuffer(raw,np.uint8).reshape(rows,-1);back=np.stack([packed&15,packed>>4],-1).reshape(rows,576).astype(np.int16)-8
   else:raw=codes.tobytes();back=np.frombuffer(raw,np.int8).reshape(rows,576)
   scales=sc.numpy().astype('<f4').tobytes();rest=torch.from_numpy(back.astype(np.float32))*torch.from_numpy(np.frombuffer(scales,'<f4').copy())[:,None]
   expected=quant(w,bits) if mode=='published_rule_transplant' else getattr(att,proj+'_proj').weight[sl].detach();assert torch.equal(rest,expected)
   payload.extend(raw);payload.extend(scales);checks+=1
 file=out/(mode+'_qk.bin');file.write_bytes(payload)
 manifest[mode]={'actual_file_bytes':file.stat().st_size,'sha256':hashlib.sha256(payload).hexdigest(),'roundtripped_projection_groups':checks,'all_values_exactly_match_dequantized_model':True}
 assert file.stat().st_size==10045530
json.dump(manifest,open(out/'packed_qk_manifest.json','w'),indent=2);print(json.dumps(manifest,indent=2))

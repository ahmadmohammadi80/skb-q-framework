from pathlib import Path
exec(compile(Path(__file__).with_name('run_full.py').read_text().split('data=load_dataset')[0],__file__,'exec'))
torch.set_num_threads(1)
tokens=tokenizer('The purpose of this numerical check is to verify the full precision rotational invariance of grouped query attention.',return_tensors='pt').input_ids
with torch.inference_mode():before=model(tokens).logits.clone()
with torch.no_grad():
 for li,layer in enumerate(model.model.layers):
  for g in range(3):
   u=bases[(li+g)%4]
   for j in range(3):
    sl=slice((g*3+j)*64,(g*3+j+1)*64);layer.self_attn.q_proj.weight[sl].copy_(u@layer.self_attn.q_proj.weight[sl])
   sl=slice(g*64,(g+1)*64);layer.self_attn.k_proj.weight[sl].copy_(u@layer.self_attn.k_proj.weight[sl])
with torch.inference_mode():after=model(tokens).logits
absolute=float((before-after).abs().max());rms=float((before-after).square().mean().sqrt());assert absolute<1e-3
json.dump({'model':name,'revision':revision,'tokens':tokens.shape[1],'max_logit_absolute_difference':absolute,'rms_logit_difference':rms,'quantized':False},open(out/'model_invariance.json','w'),indent=2)
print('full model invariance',absolute,rms,flush=True)

"""Two additional budgets on complete Pythia test corpus; no score/grid retuning."""
from pathlib import Path
s=Path(__file__).with_name('run_independent.py').read_text()
exec(compile(s[:s.index('opts=[]')]+"opts=json.load(open(p/'independent_scores.json'))\n"+s[s.index('def solve'):s.index('data=load_dataset')],__file__,'exec'))
torch.set_num_threads(4)
data=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='test');text='\n\n'.join(data['text']);ids=tok(text,return_tensors='pt').input_ids;blocks=[ids[:,i:i+512] for i in range(0,ids.shape[1]-511,512)];out={}
for budget in [24,72]:
 for mode in ['fixed','conditioned']:
  score,path=solve(mode,budget);apply(path);losses=[]
  with torch.inference_mode():
   for i,block in enumerate(blocks):
    losses.append(float(m(block,labels=block,use_cache=False).loss))
    if i%100==0:print(budget,mode,i,flush=True)
  key=f'{4+budget/24:g}bit_{mode}';out[key]={'budget_units':budget,'QK_weight_average_bits':4+budget/24,'ppl':math.exp(sum(losses)/len(losses)),'block_nll':losses,'allocation':path,'surrogate':score,'theoretical_qk_bytes':48*(128*512*4//8+128*4)+budget*64*512*4//8+48+48*128*4}
  json.dump(out,open(p/'budget_results.json','w'),indent=2);print(key,out[key]['ppl'],flush=True)

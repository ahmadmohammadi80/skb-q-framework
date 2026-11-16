"""Strong finite-grid joint competitor, using the already executed score table.
No new model PPL is claimed: identical bit/basis maps imply identical quantized weights.
"""
import json
from pathlib import Path
p=Path(__file__).resolve().parent
scores=json.load(open(p.parent/'scores.json'))
D=[0,1,3,4]
def solve(tables,budget):
 states={0:(0.0,[])}
 for table in tables:
  nxt={}
  for c,(v,path) in states.items():
   for j,w in enumerate(table):
    nc=c+D[j];nv=v+w
    if nc<=budget and (nc not in nxt or nv<nxt[nc][0]):nxt[nc]=(nv,path+[j])
  states=nxt
 return states[budget]
def joint(budget):
 bases=[[min(range(4),key=lambda k:t[k][j]) for j in range(4)] for t in scores]
 val,bits=solve([[t[bases[i][j]][j] for j in range(4)] for i,t in enumerate(scores)],budget)
 return val,[(j,bases[i][j]) for i,j in enumerate(bits)]
def alternating(budget):
 basis=[min(range(4),key=lambda k:t[k][0]) for t in scores];history=[];seen=set()
 for iteration in range(100):
  val,bits=solve([t[u] for t,u in zip(scores,basis)],budget)
  old=tuple(zip(bits,basis));history.append(val)
  newbasis=[min(range(4),key=lambda k:t[k][j]) for t,j in zip(scores,bits)]
  if newbasis==basis:return val,list(old),history
  if old in seen:raise RuntimeError('cycle')
  seen.add(old);basis=newbasis
 raise RuntimeError('no convergence')
out=[]
for budget in [90,180,270]:
 exact,em=joint(budget);alt,am,h=alternating(budget)
 out.append({'weight_mean_bits':4+budget/90,'upgrade_units':budget,'conditioned_objective':exact,'alternating_objective':alt,'identical_bit_and_basis_map':em==am,'iterations':len(h),'alternating_objective_history':h,'model_inference_performed':False,'ppl_implication':'Identical weights and identical deterministic evaluation produce identical PPL.' if em==am else 'PPL not established.'})
(p/'joint_comparator.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

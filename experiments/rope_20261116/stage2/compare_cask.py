"""Compare against the pinned, unmodified official exact CASK-Q prefix core.
Usage: python compare_cask.py /path/to/quantization-reference
"""
import sys,json,time,subprocess,hashlib
from pathlib import Path
import numpy as np,torch
from context_score import ContextScore,rotation
p=Path(__file__).parent;reference=Path(sys.argv[1]);sys.path.insert(0,str(reference/'code'))
from prefix_moment import exact_prefix_error
from cask_core import centered_causal_logits
torch.set_num_threads(1);rng=np.random.default_rng(20261117);d,n,L=8,12,32;metric=ContextScore(1/(10000**(np.arange(0,d,2)/d)),L)
q,k=rng.normal(size=(2,d,n));a=q+.12*rng.normal(size=q.shape);b=k+.12*rng.normal(size=k.shape);R=torch.tensor(np.stack([rotation(metric.freq,i) for i in range(L)]));weights=[torch.tensor(w) for w in [a,b,q,k]]
start=time.perf_counter();analytic=metric.score(q,k,a,b)*8/(L+1);analytic_seconds=time.perf_counter()-start
samples=[];max_dense_difference=0;start=time.perf_counter()
for it in range(64):
 x=torch.tensor(rng.normal(size=(32,L,n)));states=[torch.einsum('lij,blj->bli',R,x@w.T) for w in weights]
 # Preserve per-sequence values to compute an honest Monte Carlo standard error.
 for z in range(32):samples.append(float(exact_prefix_error(*[s[z] for s in states])))
 if it==0:
  dense=(centered_causal_logits(states[0],states[1])-centered_causal_logits(states[2],states[3])).square().sum()/(32*L*(L+1)/2);prefix=exact_prefix_error(*states);max_dense_difference=float(abs(dense-prefix));assert max_dense_difference<1e-10
seconds=time.perf_counter()-start;mean=float(np.mean(samples));se=float(np.std(samples,ddof=1)/np.sqrt(len(samples)));assert abs(mean-analytic)<6*se
out={'official_repository':'https://github.com/the-decentralized-compute-company/quantization','official_commit':subprocess.check_output(['git','-C',str(reference),'rev-parse','HEAD'],text=True).strip(),'official_file_sha256':hashlib.sha256((reference/'code/prefix_moment.py').read_bytes()).hexdigest(),'normalization':'expected official prefix error = 8 * analytic score / (L+1)','analytic_population_value':analytic,'official_core_monte_carlo_mean':mean,'monte_carlo_samples':len(samples),'standard_error':se,'maximum_dense_prefix_difference':max_dense_difference,'analytic_evaluation_seconds':analytic_seconds,'mc_seconds_including_input_generation':seconds,'scope':'Restricted Gaussian population objective; no claim of timing superiority for real-data empirical CASK calibration or full training reproduction.'};json.dump(out,open(p/'cask_comparison.json','w'),indent=2);print(json.dumps(out,indent=2))

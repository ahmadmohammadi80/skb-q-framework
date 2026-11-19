"""SKB-Q research candidate: EXACT finite-environment moment robust bit allocation.

Distinct from simply recalculating a Gaussian expected attention error:
uses simultaneous upper-envelope over a frozen, explicitly supplied family of
input covariance/kurtosis environments and Pareto-vector budget DP.

Analytic score valid only for two-token, independent, independent-coordinate,
zero-mean inputs. No empirical claim about real RoPE models.
See RESEARCH_NOTES.md for exact derivation, originality caveat, and limitations.
"""
from fractions import Fraction as F
from itertools import product
import random

def Q(x):return x if isinstance(x,F) else F(x)

def score(E,variance,kurtosis):
    """E[(X^T E X - X^T E Y)^2] exactly, independent X,Y.

    Assumes independent coordinates, mean zero, Var Xi=v_i>0,
    E[Xi^4]=k_i v_i^2. E may be non-symmetric.
    """
    d=len(E)
    if len(variance)!=d or len(kurtosis)!=d or any(len(row)!=d for row in E):
        raise ValueError('shape mismatch')
    if any(v<=0 for v in variance) or any(k<1 for k in kurtosis):
        raise ValueError('invalid environment')
    tr=sum((Q(E[i][i])*variance[i] for i in range(d)),F(0))
    tr_es_es=sum((Q(E[i][j])*Q(E[j][i])*variance[i]*variance[j]
                    for i in range(d) for j in range(d)),F(0))
    frob=sum((Q(E[i][j])**2*variance[i]*variance[j]
              for i in range(d) for j in range(d)),F(0))
    excess=sum(((kurtosis[i]-3)*Q(E[i][i])**2*variance[i]**2
                for i in range(d)),F(0))
    ans=tr*tr+tr_es_es+2*frob+excess
    assert ans>=0
    return ans


def enumerate_rademacher(E,variance):
    """Independent symmetric 2-point Xi= +/-sqrt(v_i), here v_i=1 only.
    Exact rational enumeration used as a reference independent of moments.
    """
    if any(v!=1 for v in variance):raise ValueError('enumeration requires unit variances')
    d=len(E)
    out=F(0)
    for x in product((-1,1),repeat=d):
        for y in product((-1,1),repeat=d):
            a=sum((Q(E[i][j])*x[i]*(x[j]-y[j]) for i in range(d) for j in range(d)),F(0))
            out+=a*a
    return out/F(2**(2*d))


def loss_vector(module,envs,bit):
    # low vs high use supplied exact error matrices, no fake quantizer!
    matrix=module['low'] if bit==0 else module['high']
    return tuple(score(matrix,env['var'],env['kurt']) for env in envs)


def dominate(a,b):
    return all(x<=y for x,y in zip(a,b))


def frontier_alloc(modules,envs,budget):
    """Exact finite multi-environment vector DP with safe Pareto pruning.

    Each module has a choice low (cost 0) / high (integer additional cost).
    Keeps nondominated partial TOTAL loss vectors for every exact cost.
    Under losses additive across modules and same environment for all modules,
    this returns a globally minimax allocation (proof by induction).
    """
    state={0:[(tuple(F(0) for _ in envs),tuple())]}
    for module in modules:
        nxt={}
        for cost,entries in state.items():
            for bits in (0,1):
                nc=cost+(module['cost'] if bits else 0)
                if nc>budget: continue
                added=loss_vector(module,envs,bits)
                for past,history in entries:
                    vec=tuple(x+y for x,y in zip(past,added))
                    bucket=nxt.setdefault(nc,[])
                    if any(dominate(v,vec) for v,_ in bucket):continue
                    bucket[:]=[(v,h) for v,h in bucket if not dominate(vec,v)]
                    bucket.append((vec,history+(bits,)))
        state=nxt
    choices=[(max(loss),loss,hist,cost)
             for cost,entries in state.items() for loss,hist in entries]
    return min(choices,key=lambda z:(z[0],z[3],z[2]))


def brute(modules,envs,budget):
    choices=[]
    for bits in product((0,1),repeat=len(modules)):
        cost=sum(m['cost']*b for m,b in zip(modules,bits))
        if cost>budget:continue
        vec=tuple(sum((loss_vector(m,envs,b)[i] for m,b in zip(modules,bits)),F(0))
                  for i in range(len(envs)))
        choices.append((max(vec),vec,bits,cost))
    return min(choices,key=lambda z:(z[0],z[3],z[2]))


def mat(a,b,c,d):return ((F(a),F(b)),(F(c),F(d)))

def high(E):return tuple(tuple(x/F(4) for x in row) for row in E)

def envs():
    return [
        {'name':'gaussian_anisotropic_A','var':(F(2),F(1,2)),'kurt':(F(3),F(3))},
        {'name':'gaussian_anisotropic_B','var':(F(1,2),F(2)),'kurt':(F(3),F(3))},
        {'name':'isotropic_rademacher','var':(F(1),F(1)),'kurt':(F(1),F(1))}
    ]


def tests():
    r=random.Random(27041)
    for _ in range(60):
        m=mat(*(r.randrange(-2,3) for _ in range(4)))
        assert score(m,(F(1),F(1)),(F(1),F(1)))==enumerate_rademacher(m,(F(1),F(1)))
        for e in envs():
            assert score(m,e['var'],e['kurt'])>=0
    assert score(mat(1,0,0,0),(F(1),F(1)),(F(3),F(3)))==4
    assert score(mat(1,0,0,0),(F(1),F(1)),(F(1),F(1)))==2
    assert score(mat(0,F(2,3),F(2,3),0),(F(1),F(1)),(F(1),F(1)))==F(8,3)
    for seed in range(50):
        rr=random.Random(seed)
        modules=[]
        for _ in range(5):
            E=mat(*(rr.randrange(-2,3) for _ in range(4)))
            modules.append({'cost':rr.choice((1,2)),'low':E,'high':high(E)})
        for budget in range(5):
            a=frontier_alloc(modules,envs(),budget);b=brute(modules,envs(),budget)
            assert a[0]==b[0],(seed,budget,a,b)
    print('PASS: 60 exact distribution-moment identities and 250 DP-vs-bruteforce budget problems')


def find_witness():
    Eset=[mat(1,0,0,0),mat(0,0,0,1),mat(0,1,1,0),
          mat(1,0,0,1),mat(1,1,1,0),mat(1,-1,-1,0),
          mat(2,0,0,0),mat(0,0,0,2)]
    all_envs=envs()
    for inds in product(range(len(Eset)),repeat=3):
        modules=[{'cost':1,'low':Eset[i],'high':high(Eset[i])} for i in inds]
        for budget in (1,2):
            robust=brute(modules,all_envs,budget)
            gaussian=brute(modules,all_envs[:1],budget)
            if robust[2]!=gaussian[2]:
                g_bits=gaussian[2]
                g_vec=tuple(sum((loss_vector(m,all_envs,b)[j] for m,b in zip(modules,g_bits)),F(0))
                            for j in range(len(all_envs)))
                if max(g_vec)>robust[0]:
                    return inds,budget,robust,gaussian,g_vec
    return None


if __name__=='__main__':
    tests()
    found=find_witness()
    assert found is not None,'No distinct allocation witness found'
    inds,budget,rob,gauss,gvec=found
    print('WITNESS modules',inds,'budget',budget,'robust bits',rob[2],
          'robust worst',str(rob[0]),'Gaussian-only bits',gauss[2],
          'Gaussian-only worst over same 3 envs',str(max(gvec)),
          'robust env scores',tuple(map(str,rob[1])))

"""Exact SKB-Q Gaussian-assumption stress test (no GPU or model checkpoint).

Constructed operator discrepancies, not quantized model measurements.
At causal context L=2 with zero RoPE frequency, independent isotropic
Rademacher tokens have covariance I, just like N(0,I), yet the self-token
fourth moments differ. This flips the ranking of two centered-logit error
operators in the proposed Gaussian-population score. Consequently replacing
"Gaussian" by "isotropic, independent" in the theorem is NOT justified.
"""
from fractions import Fraction as F
from itertools import product


def mv(m, v):
    return tuple(sum(row[j]*v[j] for j in range(2)) for row in m)


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def discrepancy(m,x,y):
    return dot(x,mv(m,y))


def gaussian_self_moment(m):
    # Wick/Isserlis for Gaussian x: E[(x^T M x)^2].
    frob=sum(v*v for row in m for v in row)
    square_trace=sum(m[i][j]*m[j][i] for i in range(2) for j in range(2))
    trace=sum(m[i][i] for i in range(2))
    return frob+square_trace+trace*trace


def gaussian_score_L2(m):
    # J = (off-diagonal second moment + self second moment)/(16*d), d=2.
    return (sum(v*v for row in m for v in row)+gaussian_self_moment(m))/32


def rademacher_score_L2(m):
    # Enumerate all two-token sequences; exact rational arithmetic.
    signs=list(product((F(-1),F(1)),repeat=2))
    return sum((discrepancy(m,x2,x1)-discrepancy(m,x2,x2))**2
               for x1 in signs for x2 in signs) / (F(len(signs)**2)*32)
    # Centered row2 gives ||H delta||^2=(a-b)^2/(2*d).
    # With L=2, J=E(a-b)^2/(16*d) = E(a-b)^2/32.


def self_test():
    for i in range(2):
        for j in range(2):
            assert sum(x[i]*x[j] for x in product((-1,1),repeat=2)) == (4 if i==j else 0)
    A=((F(1),F(0)),(F(0),F(-1)))
    B=((F(2,3),F(0)),(F(0),F(2,3)))
    assert gaussian_self_moment(A)==4
    assert gaussian_self_moment(B)==F(32,9)
    ga,gb=gaussian_score_L2(A),gaussian_score_L2(B)
    ra,rb=rademacher_score_L2(A),rademacher_score_L2(B)
    assert (ga,gb)==(F(3,16),F(5,36))
    assert (ra,rb)==(F(1,16),F(1,12))
    assert gb < ga and ra < rb  # reverse ordering despite identical covariance
    print("SKB-Q isotropic non-Gaussian stress test passed:")
    print("  Gaussian A/B:",ga,gb,"-> B")
    print("  Rademacher A/B:",ra,rb,"-> A")
    print("  Scope: mathematical counterexample; no quantization/PPL result")


if __name__=="__main__":
    self_test()

#!/usr/bin/env python3
"""Principal Component Analysis."""
import sys, math, random
random.seed(42)
def mean(v): return sum(v)/len(v)
def cov_matrix(X):
    n,d=len(X),len(X[0]); mu=[mean([X[i][j] for i in range(n)]) for j in range(d)]
    C=[[0]*d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            C[i][j]=sum((X[k][i]-mu[i])*(X[k][j]-mu[j]) for k in range(n))/(n-1)
    return C,mu
def power_iteration(M,iters=100):
    n=len(M); v=[random.gauss(0,1) for _ in range(n)]
    for _ in range(iters):
        nv=[sum(M[i][j]*v[j] for j in range(n)) for i in range(n)]
        norm=math.sqrt(sum(x*x for x in nv)); v=[x/norm for x in nv]
    eigenvalue=sum(sum(M[i][j]*v[j] for j in range(n))*v[i] for i in range(n))
    return eigenvalue,v
# Demo: 3D data with 2 main components
X=[(random.gauss(0,3),random.gauss(0,1),random.gauss(0,0.5)) for _ in range(50)]
C,mu=cov_matrix(X)
ev1,pc1=power_iteration(C)
print("PCA Results:")
print(f"  PC1: eigenvalue={ev1:.3f}, direction=[{', '.join(f'{x:.3f}' for x in pc1)}]")
print(f"  Variance explained: {ev1/sum(C[i][i] for i in range(len(C)))*100:.1f}%")

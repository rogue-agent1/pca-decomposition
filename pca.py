#!/usr/bin/env python3
"""Principal Component Analysis from scratch."""
import sys, math, random

def mean(v): return sum(v)/len(v)
def center(X):
    cols = len(X[0])
    means = [mean([r[j] for r in X]) for j in range(cols)]
    return [[r[j]-means[j] for j in range(cols)] for r in X], means

def cov_matrix(X):
    n, d = len(X), len(X[0])
    C = [[0]*d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            C[i][j] = sum(X[k][i]*X[k][j] for k in range(n))/(n-1)
    return C

def power_iteration(A, iters=100):
    n = len(A)
    v = [random.gauss(0,1) for _ in range(n)]
    norm = math.sqrt(sum(x*x for x in v))
    v = [x/norm for x in v]
    for _ in range(iters):
        Av = [sum(A[i][j]*v[j] for j in range(n)) for i in range(n)]
        norm = math.sqrt(sum(x*x for x in Av))
        v = [x/norm for x in Av]
    eigenvalue = sum(sum(A[i][j]*v[j] for j in range(n))*v[i] for i in range(n))
    return eigenvalue, v

def pca(X, n_components=2):
    Xc, means = center(X)
    C = cov_matrix(Xc)
    components = []
    for _ in range(n_components):
        val, vec = power_iteration(C)
        components.append((val, vec))
        # Deflate
        n = len(C)
        for i in range(n):
            for j in range(n):
                C[i][j] -= val * vec[i] * vec[j]
    # Project
    projected = []
    for row in Xc:
        projected.append([sum(row[j]*comp[1][j] for j in range(len(row))) for comp in components])
    total_var = sum(components[i][0] for i in range(len(components)))
    return projected, components, total_var

if __name__ == '__main__':
    random.seed(42)
    X = [[random.gauss(0,1)*3+random.gauss(0,1), random.gauss(0,1)+random.gauss(0,1)*2, random.gauss(0,0.5)] for _ in range(50)]
    proj, comps, total = pca(X, 2)
    print("PCA: 3D → 2D")
    for i, (val, vec) in enumerate(comps):
        print(f"  PC{i+1}: eigenvalue={val:.3f} ({val/sum(c[0] for c in comps)*100:.1f}%) vector={[f'{v:.3f}' for v in vec]}")
    print(f"\nFirst 5 projected points:")
    for p in proj[:5]: print(f"  [{p[0]:.3f}, {p[1]:.3f}]")

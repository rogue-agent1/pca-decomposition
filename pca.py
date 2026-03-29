#!/usr/bin/env python3
"""Principal Component Analysis from scratch."""
import sys, math

def mean_center(X):
    n, d = len(X), len(X[0])
    means = [sum(X[i][j] for i in range(n))/n for j in range(d)]
    return [[X[i][j]-means[j] for j in range(d)] for i in range(n)], means

def cov_matrix(X):
    n, d = len(X), len(X[0])
    C = [[0.0]*d for _ in range(d)]
    for i in range(d):
        for j in range(i, d):
            C[i][j] = sum(X[k][i]*X[k][j] for k in range(n))/(n-1)
            C[j][i] = C[i][j]
    return C

def power_iteration(C, n_iter=200):
    d = len(C)
    import random
    v = [random.gauss(0,1) for _ in range(d)]
    norm = math.sqrt(sum(x*x for x in v))
    v = [x/norm for x in v]
    for _ in range(n_iter):
        nv = [sum(C[i][j]*v[j] for j in range(d)) for i in range(d)]
        norm = math.sqrt(sum(x*x for x in nv))
        if norm < 1e-10: break
        v = [x/norm for x in nv]
    eigenvalue = sum(sum(C[i][j]*v[j] for j in range(d))*v[i] for i in range(d))
    return eigenvalue, v

def pca(X, n_components=2):
    Xc, means = mean_center(X)
    C = cov_matrix(Xc)
    d = len(C)
    components = []
    for _ in range(min(n_components, d)):
        val, vec = power_iteration(C)
        components.append((val, vec))
        for i in range(d):
            for j in range(d):
                C[i][j] -= val * vec[i] * vec[j]
    projected = []
    for x in Xc:
        projected.append([sum(x[j]*components[k][1][j] for j in range(d)) for k in range(len(components))])
    return projected, components, means

def test():
    import random; random.seed(42)
    X = [[i + random.gauss(0,0.1), i*2 + random.gauss(0,0.1), random.gauss(0,1)] for i in range(50)]
    proj, comps, means = pca(X, 2)
    assert len(proj) == 50
    assert len(proj[0]) == 2
    assert comps[0][0] > comps[1][0]  # first eigenvalue larger
    assert len(comps[0][1]) == 3
    print("  pca: ALL TESTS PASSED")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test": test()
    else: print("PCA from scratch")

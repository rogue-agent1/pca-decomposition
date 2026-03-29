#!/usr/bin/env python3
"""PCA — dimensionality reduction via eigendecomposition."""
import math, random, sys

def mean_center(X):
    n, d = len(X), len(X[0])
    means = [sum(X[i][j] for i in range(n))/n for j in range(d)]
    return [[X[i][j]-means[j] for j in range(d)] for i in range(n)], means

def cov_matrix(X):
    n, d = len(X), len(X[0]); C = [[0]*d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            C[i][j] = sum(X[k][i]*X[k][j] for k in range(n))/(n-1)
    return C

def power_iteration(matrix, num_iter=100):
    d = len(matrix); v = [random.gauss(0,1) for _ in range(d)]
    for _ in range(num_iter):
        mv = [sum(matrix[i][j]*v[j] for j in range(d)) for i in range(d)]
        norm = math.sqrt(sum(x**2 for x in mv))
        v = [x/norm for x in mv]
    eigenvalue = sum(sum(matrix[i][j]*v[j] for j in range(d))*v[i] for i in range(d))
    return eigenvalue, v

def pca(X, n_components=2):
    Xc, means = mean_center(X); C = cov_matrix(Xc)
    components = []; d = len(C)
    for _ in range(n_components):
        val, vec = power_iteration(C)
        components.append((val, vec))
        for i in range(d):
            for j in range(d):
                C[i][j] -= val * vec[i] * vec[j]
    return components, Xc

if __name__ == "__main__":
    random.seed(42); n = 100
    X = [[random.gauss(0,3), 0] for _ in range(n)]
    X = [[x[0]+random.gauss(0,0.5), x[0]*0.7+random.gauss(0,0.5)] for x in X]
    comps, Xc = pca(X, 2)
    print("PCA results:")
    total_var = sum(c[0] for c in comps)
    for i, (val, vec) in enumerate(comps):
        print(f"  PC{i+1}: eigenvalue={val:.3f} ({val/total_var*100:.1f}%) direction=[{vec[0]:.3f}, {vec[1]:.3f}]")

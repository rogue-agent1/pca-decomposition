#!/usr/bin/env python3
"""Principal Component Analysis from scratch."""
import sys, math, csv, random

def mean_vec(X):
    n, d = len(X), len(X[0])
    return [sum(X[i][j] for i in range(n)) / n for j in range(d)]

def center(X):
    mu = mean_vec(X)
    return [[X[i][j] - mu[j] for j in range(len(mu))] for i in range(len(X))], mu

def cov_matrix(X):
    n, d = len(X), len(X[0])
    C = [[0]*d for _ in range(d)]
    for i in range(d):
        for j in range(i, d):
            val = sum(X[k][i] * X[k][j] for k in range(n)) / (n - 1)
            C[i][j] = C[j][i] = val
    return C

def power_iteration(A, num_iter=1000):
    n = len(A)
    v = [random.gauss(0, 1) for _ in range(n)]
    norm = math.sqrt(sum(x**2 for x in v))
    v = [x/norm for x in v]
    for _ in range(num_iter):
        w = [sum(A[i][j]*v[j] for j in range(n)) for i in range(n)]
        norm = math.sqrt(sum(x**2 for x in w))
        if norm < 1e-10: break
        v = [x/norm for x in w]
    eigenvalue = sum(v[i]*sum(A[i][j]*v[j] for j in range(n)) for i in range(n))
    return eigenvalue, v

def deflate(A, eigenvalue, eigenvector):
    n = len(A)
    return [[A[i][j] - eigenvalue * eigenvector[i] * eigenvector[j] for j in range(n)] for i in range(n)]

def pca(X, n_components=2):
    Xc, mu = center(X); C = cov_matrix(Xc)
    components = []; eigenvalues = []
    A = [row[:] for row in C]
    for _ in range(n_components):
        val, vec = power_iteration(A)
        eigenvalues.append(val); components.append(vec)
        A = deflate(A, val, vec)
    total_var = sum(C[i][i] for i in range(len(C)))
    explained = [ev / total_var for ev in eigenvalues]
    projected = [[sum(Xc[i][j] * components[k][j] for j in range(len(Xc[0]))) for k in range(n_components)] for i in range(len(X))]
    return projected, components, eigenvalues, explained

def main():
    import argparse
    p = argparse.ArgumentParser(description="PCA from scratch")
    p.add_argument("file", nargs="?"); p.add_argument("-n", type=int, default=2)
    p.add_argument("--demo", action="store_true"); p.add_argument("--plot", action="store_true")
    args = p.parse_args()
    if args.demo or not args.file:
        random.seed(42)
        X = [[random.gauss(0,3) + random.gauss(0,0.5), random.gauss(0,3)*0.7 + random.gauss(0,0.5), random.gauss(0,1)] for _ in range(50)]
        proj, comps, evals, explained = pca(X, 2)
        print(f"PCA on {len(X)}x{len(X[0])} data -> {len(proj[0])} components")
        for i, (ev, exp) in enumerate(zip(evals, explained)):
            print(f"  PC{i+1}: eigenvalue={ev:.3f}, explained={exp:.1%}")
            print(f"    loadings: [{', '.join(f'{v:.3f}' for v in comps[i])}]")
        print(f"  Total explained: {sum(explained):.1%}")
        return
    with open(args.file) as f:
        reader = csv.reader(f); headers = next(reader)
        X = [[float(v) for v in row] for row in reader]
    proj, comps, evals, explained = pca(X, args.n)
    print(f"PCA: {len(X[0])}D -> {args.n}D")
    for i, exp in enumerate(explained): print(f"  PC{i+1}: {exp:.1%} variance")
    print(f"  Total: {sum(explained):.1%}")

if __name__ == "__main__": main()

import sys
import math
import numpy as np
from sklearn.metrics import silhouette_score

import symnmf  

ERR = "An Error Has Occurred"

# File IO
def read_vectors_from_txt(filename):
    try:
        X = []
        with open(filename, "r") as f:
            for line in f:
                s = line.strip()
                if s:
                    X.append([float(x) for x in s.split(",")])
        return X
    except Exception:
        return None

# K-means 
def kmeans_clusters(k, X, max_iter=300, eps=1e-4):
    N = len(X)
    if N == 0:
        raise ValueError("empty data")
    d = len(X[0])

    centroids = [X[i][:] for i in range(k)]  # first-K init

    for _ in range(max_iter):
        sums   = [[0.0]*d for _ in range(k)]
        counts = [0]*k
        labels = []

        # assignment
        for x in X:
            best = 0
            best_d2 = sum((x[j] - centroids[0][j])**2 for j in range(d))
            for c in range(1, k):
                d2 = sum((x[j] - centroids[c][j])**2 for j in range(d))
                if d2 < best_d2:
                    best_d2, best = d2, c
            labels.append(best)
            counts[best] += 1
            for j in range(d):
                sums[best][j] += x[j]

        # update + convergence
        max_shift2 = 0.0
        newc = []
        for c in range(k):
            if counts[c] > 0:
                mu = [sums[c][j] / counts[c] for j in range(d)]
            else:
                mu = centroids[c][:]
            shift2 = sum((mu[j] - centroids[c][j])**2 for j in range(d))
            if shift2 > max_shift2:
                max_shift2 = shift2
            newc.append(mu)

        centroids = newc
        if math.sqrt(max_shift2) < eps:
            break

    return labels

# SymNMF
def symnmf_clusters(k, X):
    N = len(X)
    X_np = np.array(X, dtype=float)

    if hasattr(symnmf, "calculate_symnmf_assignment"):
        labs = symnmf.calculate_symnmf_assignment(X_np, N, k)
        return list(map(int, labs)) if labs is not None else None

    if hasattr(symnmf, "calculate_symnmf_matrix"):
        H = np.array(symnmf.calculate_symnmf_matrix(X_np, N, k), dtype=float)
        return list(np.argmax(H, axis=1).astype(int))

    if hasattr(symnmf, "symnmf"):
        H = np.array(symnmf.symnmf(X_np.tolist(), k, N), dtype=float)
        return list(np.argmax(H, axis=1).astype(int))

    return None

# Silhouette (safe wrapper for 1-cluster cases) 
def safe_silhouette(X, labels):
    labels = list(map(int, labels))
    if len(set(labels)) < 2:
        return 0.0
    try:
        return silhouette_score(X, labels)
    except Exception:
        return 0.0
    
# Main function
if __name__ == "__main__":
    # args
    if len(sys.argv) != 3:
        print(ERR); sys.exit(1)
    try:
        K = int(sys.argv[1])
        filename = str(sys.argv[2])
    except Exception:
        print(ERR); sys.exit(1)
    if not filename.endswith(".txt"):
        print(ERR); sys.exit(1)

    # read data & validate K
    X = read_vectors_from_txt(filename)
    if X is None:
        print(ERR); sys.exit(1)
    N = len(X)
    if not (1 < K < N):
        print(ERR); sys.exit(1)

    try:
        labs_km  = kmeans_clusters(K, X)
        labs_nmf = symnmf_clusters(K, X)
        if labs_km is None or labs_nmf is None:
            print(ERR); sys.exit(1)

        nmf_score = safe_silhouette(X, labs_nmf)
        km_score  = safe_silhouette(X, labs_km)

        print("nmf: %.4f" % nmf_score)
        print("kmeans: %.4f" % km_score)

    except Exception:
        print(ERR); sys.exit(1)

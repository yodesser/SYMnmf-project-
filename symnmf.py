import os
import sys
import pandas as pd
import numpy as np
import math
import random
import mysymnmf as sy

def read_arguments():
    """
    Return parsed args:
      - symnmf.py: (k, goal, fileName)
      - analysis.py: (k, fileName)
    Only enforce k>1 when goal=='symnmf' (or when running analysis.py).
    """
    filePath = os.path.basename(sys.argv[0])

    # Number of arguments checkup
    if (filePath == "symnmf.py" and len(sys.argv) != 4) or \
       (filePath == "analysis.py" and len(sys.argv) != 3):
        print("An Error Has Occurred")
        return None

    # Reading k
    try:
        k = int(sys.argv[1])
    except Exception:
        print("An Error Has Occurred")
        return None

    # File name checks
    fileName = sys.argv[3] if filePath == "symnmf.py" else sys.argv[2]
    if not os.path.isfile(fileName):
        print("An Error Has Occurred")
        return None

    # For analisys.py
    if filePath == "analysis.py":
        if k <= 1:
            print("An Error Has Occurred")
            return None
        return k, fileName

    # For symnmf.py, parse goal
    goal = sys.argv[2]
    if goal not in ("sym", "ddg", "norm", "symnmf"):
        print("An Error Has Occurred")
        return None

    # Enforce k>1 for symnmf goal
    if goal == "symnmf" and k <= 1:
        print("An Error Has Occurred")
        return None

    return k, goal, fileName

# Read numeric matrix from input
def read_data(fileName):
   
    try:
        df = pd.read_csv(fileName, header=None)
        data = df.values.astype(float)
        N = data.shape[0]
        return data, N
    except Exception:
        return None, 0


def print_matrix(A):
    arr = np.asarray(A, dtype=float)
    last = len(arr) - 1
    for i, row in enumerate(arr):
        sys.stdout.write(",".join(f"{x:.4f}" for x in row))
        if i != last:
            sys.stdout.write("\n")


def calculate_symnmf_matrix(data, N, k):
    # W from C extension (list of lists) -> numpy array to compute mean
    W_list = sy.norm(data.tolist())
    if W_list is None:
        return None

    W = np.array(W_list, dtype=float)

    # H init
    np.random.seed(1234)
    m = float(W.mean())
    hi = 2.0 * math.sqrt(m / float(k)) if k > 0 else 0.0
    H0 = np.random.uniform(0.0, hi, size=(N, k))

    # Call the C extension
    H_result = sy.symnmf(H0.tolist(), W.tolist(), k)
    return np.array(H_result, dtype=float)


def calculate_symnmf_assignment(data, N, k):
    H = calculate_symnmf_matrix(data, N, k)
    if H is None:
        return None
    return np.argmax(H, axis=1).tolist()


def main():
    parsed = read_arguments()
    if parsed is None:
        return

    k, goal, fileName = parsed

    data, N = read_data(fileName)
    if data is None or N == 0:
        print("An Error Has Occurred")
        return

    # For symnmf goal only, checking k < N
    if goal == "symnmf" and k >= N:
        print("An Error Has Occurred")
        return

    try:
        if goal == "sym":
            res = sy.sym(data.tolist())
        elif goal == "ddg":
            res = sy.ddg(data.tolist())
        elif goal == "norm":
            res = sy.norm(data.tolist())
        else:  # symnmf
            res = calculate_symnmf_matrix(data, N, k)
    except Exception:
        print("An Error Has Occurred")
        return

    print_matrix(res)


if __name__ == "__main__":
    main()

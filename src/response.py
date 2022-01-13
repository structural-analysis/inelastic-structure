import scipy.linalg
import numpy as np


def solve_cholesky(ca, b):
    # Solving Cholesky by backsubstitution
    n = b.shape[0]
    x = np.zeros((n, 1))
    for i in range(n):
        sum = b[i, 0]
        for k in range(i-1, -1, -1):
            sum = sum - ca[i, k] * x[k, 0]
        x[i, 0] = sum / ca[i, i]
    for i in range(n-1, -1, -1):
        sum = x[i, 0]
        for k in range(i+1, n):
            sum = sum - ca[k, i] * x[k, 0]
        x[i, 0] = sum / ca[i, i]
    return x


def compute_displacements(analysis_type, reduced_k, reduced_f):
    if analysis_type == "static":
        ca = scipy.linalg.cholesky(reduced_k, lower=True)
        reduced_disp = solve_cholesky(ca, reduced_f)

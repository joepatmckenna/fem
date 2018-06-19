import numpy as np


def binomial_coefficients(n, k):
    seq = np.empty(k + 1, dtype=int)
    seq[0] = 1
    for i in range(1, k + 1):
        seq[i] = seq[i - 1] * (n - i + 1) / i
    return seq


def mixed_radix_to_base_10(x, m):
    """x: digits, m: bases"""
    res = x[0]
    for i in range(1, len(x)):
        res *= m[i]
        res += x[i]
    return res


def multiindices(n, deg):

    if deg > n:
        return []

    p = binomial_coefficients(n, deg)[-1]

    vars = np.empty((p, deg), dtype=int)
    var = np.arange(deg)
    vars[0] = var.copy()

    for i in range(1, p):

        idx = deg - 1
        var[idx] += 1
        while var[idx] + deg - idx > n:
            idx -= 1
            var[idx] += 1
        for idx in range(idx + 1, deg):
            var[idx] = var[idx - 1] + 1

        vars[i] = var.copy()

    return vars

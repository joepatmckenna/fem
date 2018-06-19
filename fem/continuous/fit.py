import numpy as np
from scipy.special import erf
from numpy.linalg import solve
from .. import fortran_module


def fit(x, y=None, iters=100, atol=1e-8, rtol=1e-5, impute=False):

    x = np.array(x)
    n = x.shape[0]

    if y is None:
        impute = True
        y = x.copy()

    # dt = 1.0
    # sqrt_dt = np.sqrt(dt)
    # sqrt_2 = np.sqrt(2)
    # rat = sqrt_dt / sqrt_2

    # s = np.sign(y - x)
    # mean_x = x.mean(1)
    # cov_x = np.cov(x)
    # x_mean0 = x - mean_x[:, np.newaxis]

    # w = np.empty((n, n))
    # d = []
    # for i in range(n):
    #     res = fit_i(i, x, s, cov_x, x_mean0, iters, impute)
    #     w[i] = res[0]
    #     d.append(res[1])

    # w /= rat

    w, d, it = fortran_module.fortran_module.continuous_fit(
        x, y, iters, atol, rtol, impute)

    d = [di[1:it[i]] for i, di in enumerate(d)]

    return w, d


def fit_i(i, x, s, cov_x, x_mean0, iters, impute):

    n, l = x.shape

    w = np.zeros(n)
    w[i] = 1

    erf_last = erf(x[i]) + 1

    e = []

    for it in range(iters):

        h = w.dot(x)

        erf_next = erf(h)

        ei = np.linalg.norm(erf_next - erf_last)

        e.append(ei)
        if ei < 1e-5:
            break
        erf_last = erf_next.copy()

        h *= s[i] / erf_next

        w = solve(cov_x, x_mean0.dot(h) / (l - 1))

        if impute:
            w[i] = 0

    return w, e[1:]

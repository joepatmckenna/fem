import numpy as np


def model_parameters(n, dist=None, dist_par=None):

    if (dist is None) or (dist_par is None):
        dist = np.random.uniform
        dist_par = (-0.5, 0.5)

    w = dist(*dist_par, size=(n, n))
    w[np.diag_indices_from(w)] -= 2.0
    w /= np.sqrt(n)

    return w


def time_series(w, l=None, o=1.0):

    if l is None:
        l = int(o * np.prod(w.shape))
    else:
        l = int(l)

    n = w.shape[0]
    dt = 1.0

    sqrt_dt = np.sqrt(dt)
    sqrt_2 = np.sqrt(2)
    rat = sqrt_dt / sqrt_2

    x = np.zeros((n, l))

    x[:, 0] = np.random.uniform(-1, 1, size=n)

    noise = np.random.normal(size=(n, l - 1))
    for t in range(1, l):
        x[:, t] = x[:, t - 1] + w.dot(
            x[:, t - 1]) * dt + noise[:, t - 1] * sqrt_dt

    return x

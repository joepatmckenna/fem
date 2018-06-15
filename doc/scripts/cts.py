from scipy.linalg import solve
from scipy.special import erf as sperf
import matplotlib.pyplot as plt
import numpy as np

n = 20
dt, T = 1, int(1e4)

l = np.int(np.ceil(T / dt))
sqrt_dt = np.sqrt(dt)
sqrt_2 = np.sqrt(2)

f = np.random.uniform(-0.5, 0.5, size=(n, n))
f[np.diag_indices_from(f)] -= 2.0
f /= np.sqrt(n)

x = np.zeros((n, l))
x[:, 0] = np.random.uniform(-1, 1, size=n)
noise = np.random.normal(size=(n, l - 1))
for t in range(1, l):
    x[:, t] = x[:, t - 1] + f.dot(x[:, t - 1]) * dt + noise[:, t - 1] * sqrt_dt

x1, x2 = x[:, :-1], x[:, 1:]
y = np.sign(np.diff(x))
c_j = x.mean(1)
c_jk = np.cov(x)
xc = x1 - c_j[:, np.newaxis]


def fit(i, iters=10):

    fi = np.random.uniform(-0.5, 0.5, size=(1, n))
    fi[0, i] -= 2

    sperf_last = sperf(fi.dot(x1) * sqrt_dt / sqrt_2)

    for it in range(iters):

        h = fi.dot(x1)

        h *= y[i] / sperf(h * sqrt_dt / sqrt_2)

        print c_jk.shape, h.shape, xc.shape

        fi = solve(c_jk, (h * xc).mean(1))

        sperf_next = sperf(fi.dot(x1) * sqrt_dt / sqrt_2)
        e = np.linalg.norm(sperf_next - sperf_last)
        print i, it, e
        if e * e < 1e-4:
            break
        sperf_last = sperf_next.copy()

    return fi


f_fit = np.empty((n, n))
for i in range(n):
    f_fit[i] = fit(i)

f_flat = f.flatten()
f_fit_flat = f_fit.flatten()
plt.scatter(f_flat, f_fit_flat, c='k', s=0.1)
plt.show()

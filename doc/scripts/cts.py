from scipy.linalg import solve
from scipy.special import erf as sperf
import matplotlib.pyplot as plt
import numpy as np

n = 20
dt, T = 1., int(1e4)

l = np.int(np.ceil(T / dt))
sqrt_dt = np.sqrt(dt)
sqrt_2 = np.sqrt(2)
rat = sqrt_dt / sqrt_2

w = np.random.uniform(-0.5, 0.5, size=(n, n))
w[np.diag_indices_from(w)] -= 2.0
w /= np.sqrt(n)

x = np.zeros((n, l))
x[:, 0] = np.random.uniform(-1, 1, size=n)
noise = np.random.normal(size=(n, l - 1))
for t in range(1, l):
    x[:, t] = x[:, t - 1] + w.dot(x[:, t - 1]) * dt + noise[:, t - 1] * sqrt_dt

plt.figure(figsize=(16, 4))
plt.plot(x[:, -100:].T)
plt.show()

x1, x2 = x[:, :-1], x[:, 1:]
y = np.sign(np.diff(x))
c_j = x.mean(1)
c_jk = np.cov(x)
xc = x1 - c_j[:, np.newaxis]


def fit(i, iters=100):

    wi = np.zeros(n)
    wi[i] = 1

    # sperf_last = sperf(x1[i] * rat) + 1
    sperf_last = sperf(x1[i]) + 1

    e = []

    for it in range(iters):

        h = wi.dot(x1)

        # sperf_next = sperf(h * rat)
        sperf_next = sperf(h)
        ei = np.linalg.norm(sperf_next - sperf_last)
        e.append(ei)
        print i, it, ei
        if ei * ei < 1e-5:
            break
        sperf_last = sperf_next.copy()

        h *= y[i] / sperf_next

        wi = solve(c_jk, xc.dot(h) / (l - 1))

    return wi, e


w_fit = np.empty((n, n))
e = []
for i in range(n):
    res = fit(i)
    w_fit[i] = res[0] / rat
    e.append(res[1])

w_flat = w.flatten()
w_fit_flat = w_fit.flatten()
plt.scatter(w_flat, w_fit_flat, c='k', s=0.1)
grid = np.linspace(w_flat.min(), w_flat.max())
plt.plot(grid, grid, 'r--', lw=0.5)
plt.show()

for ei in e:
    plt.plot(ei)
plt.show()

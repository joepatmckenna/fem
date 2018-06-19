import matplotlib.pyplot as plt
from numpy import matlib
from scipy.sparse.linalg import svds
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
import multiprocessing as mp
import time

start = time.time()

m, n = 3, 10
l = int(20.0 * m * n * m * n)

dist = np.random.normal
dist_par = (0.0, 1.0 / np.sqrt(m * n))

w_size = (m * n, m * n)
w = dist(*dist_par, size=w_size)
w = np.vstack([wi - wi.mean(0) for wi in np.split(w, n)])
w = np.hstack(
    [wj - wj.mean(1)[:, np.newaxis] for wj in np.split(w, n, axis=1)])
for i in range(n):
    i1, i2 = m * i, m * (i + 1)
    w[i1:i2, i1:i2] = 0

mm = m * np.arange(n)

x = np.random.randint(m, size=(n, l), dtype=int)
s_x = lil_matrix((m * n, l))
for i in range(l):
    s_x[x[:, i] + mm, i] = 1

y = np.empty((n, l), dtype=int)
s_y = lil_matrix((m * n, l))

for i in range(l):

    h = w * s_x[:, i]

    p = np.exp(h)
    p = np.array([pi.cumsum() for pi in np.split(p, n)])
    p /= p[:, -1, np.newaxis]

    u = np.random.uniform(size=(n, 1))
    y[:, i] = (p < u).sum(1)

    s_y[y[:, i] + mm, i] = 1

s_x = s_x.tocsr()
s_y = s_y.tocsr()

s_x_svd = svds(s_x, k=m * n - n + 1)

s_x_pinv_ = [
    np.matrix(s_x_svd[2].T), (1.0 / s_x_svd[1]),
    np.matrix(s_x_svd[0].T)
]


def s_x_pinv(x):
    x = x * s_x_pinv_[0]
    x = np.multiply(x, s_x_pinv_[1])
    x = x * s_x_pinv_[2]
    return x


def fit_i((i, max_iter)):

    i1, i2 = i * m, (i + 1) * m
    s_y_i = s_y[i1:i2]

    wi = matlib.zeros((m, m * n))
    dw = wi - w[i1:i2]
    e = [np.multiply(dw, dw).mean()]

    h = wi * s_x
    p = np.exp(h)
    p /= p.sum(0)
    d = [np.power(s_y_i - p, 2).mean()]

    for it in range(1, max_iter):

        h += s_y_i - p

        wi = s_x_pinv(h)
        wi[:, i1:i2] = 0
        wi -= wi.mean(0)
        for j in range(n):
            j1, j2 = j * m, (j + 1) * m
            wi[:, j1:j2] -= wi[:, j1:j2].mean(1)
        dw = wi - w[i1:i2]
        e.append(np.multiply(dw, dw).mean())

        h = wi * s_x
        p = np.exp(h)
        p /= p.sum(0)
        d.append(np.power(s_y_i - p, 2).mean())

        if d[-1] > d[-2] or np.isclose(d[-1], d[-2]):
            break

    print i, it, e[-1], d[-1]

    return wi, e, d


max_iter = 100

args = [(i, max_iter) for i in range(n)]

pool = mp.Pool(processes=mp.cpu_count())
res = pool.map_async(fit_i, args)
res.wait()
res = res.get()
pool.close()
pool.terminate()
pool.join()

# res = [fit_i(arg) for arg in args]

print time.time() - start

w_fit = np.vstack([r[0] for r in res])
e = [r[1] for r in res]
d = [r[2] for r in res]

# fig, ax = plt.subplots(2, 1, figsize=(4, 8))
# for ei in e:
#     ax[0].plot(ei)
# for di in d:
#     ax[1].plot(di)
# plt.show()
# plt.close()

fig, ax = plt.subplots(2, 2, figsize=(8, 8))
ax[0, 1].matshow(w_fit)
for ei in e:
    ax[0, 0].plot(ei)
for di in d:
    ax[1, 0].plot(di)
lo, hi = w.min(), w.max()
ax[1, 1].plot(
    np.linspace(lo, hi), np.linspace(lo, hi), 'r-', lw=0.5, alpha=0.5)
ax[1, 1].scatter(
    w.flatten(), np.array(w_fit).flatten().squeeze(), c='k', s=0.1)
plt.show()
plt.close()

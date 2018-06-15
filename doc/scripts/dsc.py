from scipy.linalg import solve
import numpy as np
import matplotlib.pyplot as plt
from numpy import matlib
from scipy.sparse.linalg import svds
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix

m, n = 3, 20
l = int(1.0 * m * n * m * n)

dist = np.random.normal
dist_par = (0.0, 1.0 / np.sqrt(m * n))

f_size = (m * n, m * n)
f = dist(*dist_par, size=f_size)
f = np.vstack([fi - fi.mean(0) for fi in np.split(f, n)])
f = np.hstack(
    [fj - fj.mean(1)[:, np.newaxis] for fj in np.split(f, n, axis=1)])

mm = m * np.arange(n)

states = np.empty((n, l), dtype=int)
states[:, 0] = np.random.randint(m, size=n)

x = lil_matrix((m * n, l))
x[states[:, 0] + mm, 0] = 1

for t in range(1, l):

    h = f * x[:, t - 1]

    p = np.exp(h)
    p = np.array([pi.cumsum() for pi in np.split(p, n)])
    p /= p[:, -1, np.newaxis]

    u = np.random.uniform(size=(n, 1))
    states[:, t] = (p < u).sum(1)

    x[states[:, t] + mm, t] = 1

x = x.toarray()
x1, x2 = x[:, :-1], x[:, 1:]
c_j = x.mean(1)
c_jk = np.cov(x)
xc = x1 - c_j[:, np.newaxis]


def fit(i, iters=10):

    s1i = s1[i * m:(i + 1) * m]
    ds = s1i - s1i.mean(1)[:, np.newaxis]

    fi = np.random.uniform(size=(m, m * n))

    for it in range(iters):

        h = fi.dot(x1)
        p = np.exp(h)
        p /= p.sum(0)

        h *= ds / p

        for j in range(m):
            fi[j] = solve(c_jk, (h[j] * xc).mean(1))

        print i, it, np.linalg.norm(fi - f[i * m:(i + 1) * m])

    return fi


tmp = fit(0)

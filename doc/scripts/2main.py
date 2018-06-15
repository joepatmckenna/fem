import matplotlib.pyplot as plt
from numpy import matlib
from scipy.sparse.linalg import svds
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
import multiprocessing as mp

m, n = 3, 10
l = int(1.0 * m * n * m * n)

dist = np.random.normal
dist_par = (0.0, 1.0 / np.sqrt(m * n))

w_size = (m * n, m * n)
w = dist(*dist_par, size=w_size)
w = np.vstack([wi - wi.mean(0) for wi in np.split(w, n)])
w = np.hstack(
    [wj - wj.mean(1)[:, np.newaxis] for wj in np.split(w, n, axis=1)])

mm = m * np.arange(n)

states = np.empty((l, n), dtype=int)
states[0] = np.random.randint(m, size=n)

s = lil_matrix((l, m * n))
s[0, states[0] + mm] = 1

for t in range(1, l):

    h = s[t - 1] * w

    p = np.exp(h)
    p = np.array([pi.cumsum() for pi in np.split(p, n, axis=1)])
    p /= p[:, -1, np.newaxis]

    u = np.random.uniform(size=(n, 1))
    states[t] = (p < u).sum(1)

    s[t, states[t] + mm] = 1

# s = s.tocsc()
s = s.toarray()

# import numpy as np

# np.random.seed(1)

# # parameters:
# n = 100
# g = 1.0
# fL = 1.0
# g = float(g)
# fL = float(fL)
# L0 = int(fL * n**2) + 1

# L = L0 - 1

# ##=============================================================================
# # generate sequences:
# W0_all = np.random.normal(0.0, g / np.sqrt(n), size=(n, n))

# s = np.ones((L0, n))
# for t in range(L):
#     for i in range(n):
#         H = np.sum(W0_all[i, :] * s[t, :])
#         if (np.exp(-H) / (np.exp(H) + np.exp(-H))) > np.random.rand():
#             s[t + 1, i] = -1.
##=============================================================================
## corvariance:
ds = s - s.mean(0)
C = np.empty((n * m, n * m))
for i in range(n * m):
    for j in range(n * m):
        C[i, j] = np.mean(ds[:, i] * ds[:, j])
C_inv = np.linalg.inv(C)  # inverse matrix

s2 = s[1:].copy()
##=============================================================================
nloop = 1000

## predict W:
W_all = np.empty((n*m, n*m))

for i0 in range(n):

    i1, i2 = m * i0, m * (i0 + 1)

    W0 = W0_all[i1:i2, :]

    s2i = s2[:, i1:i2]

    H = s2i.copy()  # initial value

    cost = np.zeros(nloop + 1)

    W1 = np.empty((nloop, n))

    Hs = np.empty(n*m)

    iloop = 1

    stop_iloop = 0

    while iloop < nloop and stop_iloop == 0:
        for i in range(n*m):
            Hs[i] = np.mean((H[0:L] - np.mean(H)) * ds[0:L, i])

        W = np.dot(Hs[0:n], C_inv[0:n, 0:n])
        H[0:L] = np.dot(s[0:L, 0:n], W[0:n])
        cost[iloop] = np.mean((s2i - np.tanh(H))**2)
        MSE = np.mean((W0[:] - W[:])**2)

        if cost[iloop] > cost[iloop - 1] and iloop > 1:
            stop_iloop = 1

        H[:] = s2i[:] * H[:] / np.tanh(H[:])
        W1[iloop, :] = W[:]

        #print(i0,iloop,MSE,cost[iloop])
        iloop += 1

    niter = iloop - 2
    print('i0:', i0, 'niter:', niter)
    W = W1[niter, :]
    W_all[i0, :] = W[0:n]

##=============================================================================
MSE = np.mean((W0_all - W_all)**2)
slope = np.sum(W0_all * W_all) / np.sum(W0_all**2)
print(float(L) / (n**2), MSE, slope)
MSE_out.write("%f %f %f %f \n" % (g, float(L) / (n**2), MSE, slope))

for i in range(n):
    for j in range(n):
        W_out.write("%i %i %f %f \n" % (i + 1, j + 1, float(W0_all[i, j]),
                                        float(W_all[i, j])))
W_out.close()

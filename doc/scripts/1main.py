##=============================================================================
## Danh-Tai Hoang (NIDDK/NIH)
## Network reconstruction
##=============================================================================
import numpy as np
import sys

np.random.seed(1)

# parameters:
N = 100
g = 1.0
fL = 1.0
print(N, g, fL)
g = float(g)
fL = float(fL)
L0 = int(fL * N**2) + 1

L = L0 - 1

##=============================================================================
# generate sequences:
W0_all = np.random.normal(0.0, g / np.sqrt(N), size=(N, N))

s = np.ones((L0, N))
for t in range(L):
    for i in range(N):
        H = np.sum(W0_all[i, :] * s[t, :])
        if (np.exp(-H) / (np.exp(H) + np.exp(-H))) > np.random.rand():
            s[t + 1, i] = -1.
##=============================================================================
# empirical value:
m = np.mean(s[0:L, :], axis=0)

## corvariance:
ds = s - np.mean(s, axis=0)
C = np.empty((N, N))
for i in range(N):
    for j in range(N):
        C[i, j] = np.mean(ds[:, i] * ds[:, j])
C_inv = np.linalg.inv(C)  # inverse matrix

s1 = np.copy(s[1:])

##=============================================================================
nloop = 1000

## predict W:
W_all = np.empty((N, N))
for i0 in range(N):
    W0 = W0_all[i0, :]
    s11 = s1[0:L, i0]
    H = s11.copy()  # initial value
    cost = np.zeros(nloop + 1)
    W1 = np.empty((nloop, N))
    Hs = np.empty(N)
    iloop = 1
    stop_iloop = 0
    while iloop < nloop and stop_iloop == 0:
        for i in range(N):
            Hs[i] = np.mean((H[0:L] - np.mean(H)) * ds[0:L, i])

        W = np.dot(Hs[0:N], C_inv[0:N, 0:N])
        H[0:L] = np.dot(s[0:L, 0:N], W[0:N])
        cost[iloop] = np.mean((s11 - np.tanh(H))**2)
        MSE = np.mean((W0[:] - W[:])**2)

        if cost[iloop] > cost[iloop - 1] and iloop > 1:
            stop_iloop = 1

        H[:] = s11[:] * H[:] / np.tanh(H[:])
        W1[iloop, :] = W[:]

        #print(i0,iloop,MSE,cost[iloop])
        iloop += 1

    niter = iloop - 2
    print('i0:', i0, 'niter:', niter)
    W = W1[niter, :]
    W_all[i0, :] = W[0:N]

##=============================================================================
MSE = np.mean((W0_all - W_all)**2)
slope = np.sum(W0_all * W_all) / np.sum(W0_all**2)
print(float(L) / (N**2), MSE, slope)
MSE_out.write("%f %f %f %f \n" % (g, float(L) / (N**2), MSE, slope))

for i in range(N):
    for j in range(N):
        W_out.write("%i %i %f %f \n" % (i + 1, j + 1, float(W0_all[i, j]),
                                        float(W_all[i, j])))
W_out.close()

import numpy as np
import numpy.linalg as nplin
import scipy as sp
from scipy.special import erf as sperf
from scipy.linalg import pinv as spinv
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import multiprocessing
import os
import sys

dt, T, N_var = 0.1, 30000, 30  #float(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3])
# seed = 23
# np.random.seed(seed)
if len(sys.argv) > 4:
    seed = int(sys.argv[4])
    np.random.seed(seed)
L = np.int(np.ceil(T / dt))
dts = np.sqrt(dt)
sqrt2 = np.sqrt(2)


def gen_W(nvar=N_var):
    """coupling matrix"""
    #    return scale*((np.array(np.random.rand(nvar*nvar)).reshape(nvar,nvar))-0.5)-np.eye(nvar)
    scale = 1.0 / np.sqrt(float(nvar))
    return scale * (np.random.rand(nvar, nvar) - 0.5 - 2 * np.eye(nvar))


def gen_X(w, nvar=N_var, ll=L):
    """time series"""
    x = np.zeros((ll, nvar))
    eta = np.random.randn(ll - 1, nvar)
    x[0] = 2 * np.array(np.random.rand(nvar) - 0.5).reshape(1, nvar)
    for t in range(1, ll):
        x[t] = x[t - 1] + dt * np.dot(x[t - 1], w) + dts * eta[t - 1]
    return x


def gen_X_trial(w, x, nvar=N_var, ll=L):
    """time series trial"""
    X = np.zeros(ll, nvar)
    eta = np.random.randn((ll - 1, nvar))
    X[0] = x[0]
    for t in range(1, ll):
        X[t] = x[t - 1] + dt * np.dot(x[t - 1], w) + dts * eta[t - 1]
    return X


W = gen_W()
X = gen_X(W)
Y = np.sign(X[1:] - X[:-1])  #X(t+1)-X(t)
C_j = np.mean(X, axis=0)
XC = (X - C_j)[:-1]
C_jk = np.cov(X, rowvar=False)


def iteration(index, w_in, x, xc, y, num_iter=10):
    sperf_init = sperf(np.dot(x, w_in)[:-1, index] * dts / sqrt2)
    for iterate in range(num_iter):
        h = np.dot(x, w_in[:, index])[:-1]
        h_ratio = h * y[:, index] / sperf(h * dts / sqrt2)
        w_in[:, index] = sp.linalg.solve(C_jk,
                                         np.mean(
                                             h_ratio[:, np.newaxis] * xc,
                                             axis=0))
        sperf_next = sperf(np.dot(x, w_in)[:-1, index] * dts / sqrt2)
        if (nplin.norm(sperf_next - sperf_init)**2 < 1e-4): break
        sperf_init = np.copy(sperf_next)


#        print(iterate,nplin.norm((x[1:,index]-x[:-1,index])-sperf_init)**2/float(L-1))
#    return w_in

cov_plus = np.cov(X[:-1], X[1:], rowvar=False)
#w_try = sp.linalg.solve(cov_plus[:N_var,:N_var]*dt,cov_plus[:N_var,N_var:]-cov_plus[:N_var,:N_var])#gen_W()

w_try = gen_W()
print('initial MSE', nplin.norm(W - w_try)**2 / float(N_var**2))
for index in range(N_var):
    iteration(index, w_try, X, XC, Y)
    print('final MSE for ',index,nplin.norm(W[:,index]-w_try[:,index])**2/float(N_var),\
          nplin.norm(Y[:,index]-sperf(np.dot(X,w_try)[:-1,index])*dts/sqrt2)**2/float(L-1))

plt.scatter(W.flatten(), w_try.flatten(), c='k', s=0.1)
plt.show()

# with PdfPages('langevin-' + str(dt) + '-' + str(T) + '-' + str(N_var) + '-' +
#               str(seed) + '.pdf') as pdf:
#     fig = plt.figure()
#     plt.imshow(w_try)
#     plt.colorbar()
#     pdf.savefig(fig)
#     plt.close()
#     fig = plt.figure()
#     print('initial MSE', nplin.norm(W - w_try)**2 / float(N_var**2))
#     for index in range(N_var):
#         iteration(index, w_try, X, XC, Y)
#         print('final MSE for ',index,nplin.norm(W[:,index]-w_try[:,index])**2/float(N_var),\
#               nplin.norm(Y[:,index]-sperf(np.dot(X,w_try)[:-1,index])*dts/sqrt2)**2/float(L-1))

# ##    X_try = gen_X(w_try2)
# ##    fig=plt.figure()
# ##    plt.imshow(np.cov(X[:-1],X[1:],rowvar=False)-np.cov(X_try[:-1],X_try[1:],rowvar=False))
# ##    plt.colorbar()
# ##    pdf.savefig(fig)
# ##    plt.close()
# ##    X_try = gen_X(w_try)
# ##    fig=plt.figure()
# ##    plt.imshow(np.cov(X[:-1],X[1:],rowvar=False)-np.cov(X_try[:-1],X_try[1:],rowvar=False))
# ##    plt.colorbar()
# ##    pdf.savefig(fig)
# ##    plt.close()
#     fig = plt.figure()
#     plt.imshow(W)
#     plt.colorbar()
#     pdf.savefig(fig)
#     plt.close()
#     fig = plt.figure()
#     plt.imshow(w_try)
#     plt.colorbar()
#     pdf.savefig(fig)
#     plt.close()
#     fig = plt.figure()
#     plt.imshow(W - w_try)
#     plt.colorbar()
#     pdf.savefig(fig)
#     plt.close()
#     fig = plt.figure()
#     plt.imshow(C_jk)
#     plt.colorbar()
#     pdf.savefig(fig)
#     plt.close()
#     fig = plt.figure()
#     plt.imshow(np.cov(gen_X(w_try), rowvar=False))
#     plt.colorbar()
#     pdf.savefig(fig)
#     plt.close()
#     fig = plt.figure()
#     plt.imshow(C_jk - np.cov(gen_X(w_try), rowvar=False))
#     plt.colorbar()
#     pdf.savefig(fig)
#     plt.close()
#     for index in range(3):  #not N_var, too many graphs!
#         fig = plt.figure()
#         plt.plot(np.arange(100), Y[:100, index], 'b-')
#         #    plt.plot(np.arange(100),X[1:101,1],'r-')
#         plt.plot(np.arange(100), (X[1:] - X[:-1])[:100, index], 'g-')
#         plt.plot(np.arange(100), sperf(np.dot(X, w_try)[:100, index]), 'k-')
#         pdf.savefig(fig)
#         plt.close()

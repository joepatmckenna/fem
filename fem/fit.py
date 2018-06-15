import combinatorics
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
from fortran_module import fortran_module


def one_hot(x, degs):

    x = np.array(x)
    dim = len(x.shape)

    if dim == 1:
        n = 1
        m = np.array([np.unique(x).shape[0]])
        l = x.shape[0]
    elif dim == 2:
        n = x.shape[0]
        m = np.array([np.unique(xi).shape[0] for xi in x])
        l = x.shape[1]

    degs = np.array(degs)
    k = len(degs)
    max_deg = degs.max()

    idx_len = combinatorics.binomial_coefficients(n, max_deg)[degs].sum()

    idx = []
    for deg in degs:
        for i in combinatorics.multiindices(n, deg):
            idx.append(i)

    mi = np.array([np.prod(m[i]) for i in idx])

    s = np.vstack(
        [combinatorics.mixed_radix_to_base_10(x[i], m[i]) for i in idx])

    stratifier = np.insert(mi.cumsum(), 0, 0)[:-1]

    data = np.ones(idx_len * l)
    indices = (s + stratifier[:, np.newaxis]).T.flatten()
    indptr = idx_len * np.arange(l + 1)

    return csc_matrix((data, indices, indptr)), idx


def categorize(x):

    dim = x.shape
    if len(dim) == 1:
        x = np.array([x])
    elif len(dim) == 2:
        x = np.array(x)
    else:
        print 'x should be 1- or 2-dimensional'
        return

    l = x.shape[1]

    x_cat = np.empty(shape=x.shape, dtype=int)

    cat = []
    for i in range(x.shape[0]):
        unique_states = np.sort(np.unique(x[i]))
        m = len(unique_states)
        num = dict(zip(unique_states, np.arange(m)))
        for j in range(l):
            x_cat[i, j] = num[x[i, j]]
        cat.append(dict(zip(np.arange(m), unique_states)))

    return x_cat, cat


def discrete_fit(x, y, degs, iters, overfit):
    # x: sum(p) by l
    # ------------------------------------
    # x1: x[i_x[0]:i_x[1], :] -- p[0] by l
    # ------------------------------------
    # x2: x[i_x[1]:i_x[2], :] -- p[1] by l
    # ------------------------------------
    # ...
    # ------------------------------------
    # i_x = np.insert(p.cumsum(), 0, 0)

    x = np.array(x)
    y = np.array(y)

    x, cat_x = categorize(x)
    y, cat_y = categorize(y)

    m_x = np.array([len(c) for c in cat_x])
    m_y = np.array([len(c) for c in cat_y])

    n_x = x.shape[0]
    n_y = y.shape[0]

    x_oh, idx = one_hot(x, degs)

    x_oh_rank = np.linalg.matrix_rank(x_oh.todense())
    # x_oh_svd = svds(x_oh, k=min(x_oh_rank, n_x - 1))
    x_oh_svd = svds(x_oh, k=x_oh_rank)

    x_oh_pinv = [x_oh_svd[2].T, 1.0 / x_oh_svd[1], x_oh_svd[0].T]

    par, disc, it = fortran_module.discrete_fit(x, y, m_x, m_y,
                                                m_y.sum(), degs, x_oh_pinv[0],
                                                x_oh_pinv[1], x_oh_pinv[2],
                                                iters, overfit)

    idx_by_deg = [combinatorics.multiindices(n_x, deg) for deg in degs]
    mi = np.array(
        [np.sum([np.prod(m_x[i]) for i in idx]) for idx in idx_by_deg])
    mi = np.insert(mi.cumsum(), 0, 0)

    par = {deg: par[:, mi[i]:mi[i + 1]] for i, deg in enumerate(degs)}

    disc = [d[1:it[i]] for i, d in enumerate(disc)]

    return par, disc


def fit(x, y, degs=[1], iters=100, overfit=True, mode=None):

    if x.dtype != 'float' and y.dtype != 'float':
        if mode == 'impute':
            return discrete_impute_fit(x, y, degs, iters, overfit)
        else:
            return discrete_fit(x, y, degs, iters, overfit)

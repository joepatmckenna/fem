import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
import combinatorics
from .. import fortran_module


def one_hot(x, degs):
    """One hot encoding of `x`

    Args:
        x (ndarray):
        degs (list):

    Returns
        (csc_matrix, ndarray): the one hot encoding and the multiindices

    """

    x = np.array(x)

    if x.ndim == 1:
        n = 1
        m = np.array([np.unique(x).shape[0]])
        l = x.shape[0]
    elif x.ndim == 2:
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
    """Convert x to integer data

    Args:
        x (list):

    Returns:
        (list, dict): The integer data and the map from symbols to integers

    """

    n = len(x)
    l = [len(xi) for xi in x]

    # cat_x = np.empty(shape=x.shape, dtype=int)
    cat_x = [np.empty(shape=l[i], dtype=int) for i in range(n)]

    cat = []
    for i in range(n):
        unique_states = np.sort(np.unique(x[i]))
        m = len(unique_states)
        num = dict(zip(unique_states, np.arange(m)))
        for j in range(l[i]):
            cat_x[i][j] = num[x[i][j]]
        cat.append(num)

    if np.allclose(l, l[0]):
        cat_x = np.array(cat_x)

    return cat_x, cat


def fit(x, y=None, degs=[1], iters=100, overfit=True, impute=None):
    """Fit the Potts model to the data

    Args:
        x (ndarray):
        y (ndarray):
        degs (list):
        iters (int):
        overfit (bool):
        impute (bool):

    Returns:
        (dict, list): The fitted model parameters and the running discrepancies

    """

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
    x, cat_x = categorize(x)
    m_x = np.array([len(c) for c in cat_x])

    if y is None:
        impute = True
        y = x.copy()
        m_y = m_x.copy()
    else:
        impute = False
        y = np.array(y)
        y, cat_y = categorize(y)
        m_y = np.array([len(c) for c in cat_y])

    n_x, n_y = x.shape[0], y.shape[0]

    x_oh, idx = one_hot(x, degs)

    x_oh_rank = np.linalg.matrix_rank(x_oh.todense())
    x_oh_svd = svds(x_oh, k=min(x_oh_rank, min(x_oh.shape) - 1))
    # x_oh_svd = svds(x_oh, k=x_oh_rank)

    sv_pinv = x_oh_svd[1]
    zero_sv = np.isclose(sv_pinv, 0)
    sv_pinv[~zero_sv] = 1.0 / sv_pinv[~zero_sv]
    sv_pinv[zero_sv] = 0.0
    x_oh_pinv = [x_oh_svd[2].T, sv_pinv, x_oh_svd[0].T]

    w, d, it = fortran_module.fortran_module.discrete_fit(
        x, y, m_x, m_y,
        m_y.sum(), degs, x_oh_pinv[0], x_oh_pinv[1], x_oh_pinv[2], iters,
        overfit, impute)

    idx_by_deg = [combinatorics.multiindices(n_x, deg) for deg in degs]
    mi = np.array(
        [np.sum([np.prod(m_x[i]) for i in idx]) for idx in idx_by_deg])
    mi = np.insert(mi.cumsum(), 0, 0)

    w = {deg: w[:, mi[i]:mi[i + 1]] for i, deg in enumerate(degs)}

    d = [di[1:it[i]] for i, di in enumerate(d)]

    return w, d

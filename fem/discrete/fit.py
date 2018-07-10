import time
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
import combinatorics
from .. import fortran_module


def one_hot(x, m, degs):
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
        l = x.shape[0]
    elif x.ndim == 2:
        n = x.shape[0]
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
    m_sum = mi.sum()

    s = np.vstack(
        [combinatorics.mixed_radix_to_base_10(x[i], m[i]) for i in idx])

    stratifier = np.insert(mi.cumsum(), 0, 0)[:-1]

    data = np.ones(idx_len * l)
    indices = (s + stratifier[:, np.newaxis]).T.flatten()
    indptr = idx_len * np.arange(l + 1)

    return csc_matrix((data, indices, indptr), shape=(m_sum, l)), idx


def categorize(x):
    """Convert x to integer data

    Args:
        x (list):

    Returns:
        (list, dict): The integer data and the map from symbols to integers

    """

    n = len(x)
    l = [len(xi) for xi in x]

    x_int = [np.empty(shape=l[i], dtype=int) for i in range(n)]

    cat_x = []
    for i in range(n):
        unique_states = np.sort(np.unique(x[i]))
        m = len(unique_states)
        num = dict(zip(unique_states, np.arange(m)))
        for j in range(l[i]):
            x_int[i][j] = num[x[i][j]]
        cat_x.append(num)

    if np.allclose(l, l[0]):
        x_int = np.array(x_int)

    return x_int, cat_x


def fit(x, y=None, iters=100, degs=[1], overfit=True, impute=None):
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

    x_oh, idx_x = one_hot(x, m_x, degs)

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
    mm_x = np.array(
        [np.sum([np.prod(m_x[i]) for i in idx]) for idx in idx_by_deg])
    mm_x = np.insert(mm_x.cumsum(), 0, 0)

    w = {deg: w[:, mm_x[i]:mm_x[i + 1]] for i, deg in enumerate(degs)}
    d = [di[1:it[i]] for i, di in enumerate(d)]

    return w, d


class model:
    def __init__(self, degs=[1]):
        self.degs = degs
        # x, y, n_x, n_y, m_x, m_y, cat_x, cat_y, x_oh_pinv
        # w, d, degs, impute

    def fit(self,
            x,
            y=None,
            iters=100,
            overfit=True,
            impute=None,
            svd='approx'):
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

        degs = self.degs

        x = np.array(x)
        x_int, cat_x = categorize(x)

        m_x = np.array([len(c) for c in cat_x])
        n_x = x_int.shape[0]

        if y is None:
            impute = True
            y = x
            y_int, cat_y = x_int, cat_x
            m_y = m_x
            n_y = n_x
        else:
            impute = False
            y = np.array(y)
            y_int, cat_y = categorize(y)
            m_y = np.array([len(c) for c in cat_y])
            n_y = y_int.shape[0]

        cat_x_inv = [{v: k for k, v in cat.iteritems()} for cat in cat_x]
        cat_y_inv = [{v: k for k, v in cat.iteritems()} for cat in cat_y]

        m_x_cumsum = np.insert(m_x.cumsum(), 0, 0)
        m_y_cumsum = np.insert(m_y.cumsum(), 0, 0)

        idx_x_by_deg = [combinatorics.multiindices(n_x, deg) for deg in degs]
        mm_x = np.array(
            [np.sum([np.prod(m_x[i]) for i in idx]) for idx in idx_x_by_deg])
        mm_x_cumsum = np.insert(mm_x.cumsum(), 0, 0)

        if (not impute) or (impute and svd == 'approx'):

            x_oh = one_hot(x_int, m_x, degs)[0]
            x_oh_pinv = svd_pinv(x_oh)

            w, d, it = fortran_module.fortran_module.discrete_fit(
                x_int, y_int, m_x, m_y,
                m_y.sum(), degs, x_oh_pinv[0], x_oh_pinv[1], x_oh_pinv[2],
                iters, overfit, impute)

            d = [di[1:it[i]] for i, di in enumerate(d)]

        elif impute and svd == 'exact':

            w, d = np.zeros((mm_x_cumsum[-1], mm_x_cumsum[-1])), []

            for i in range(n_x):

                not_i = np.delete(range(n_x), i)
                x_oh = one_hot(x_int[not_i], m_x[not_i], degs)[0]

                x_oh_pinv = svd_pinv(x_oh)

                wi, di, it = fortran_module.fortran_module.discrete_fit(
                    x_int[not_i], y_int[[i]], m_x[not_i], m_y[[i]],
                    m_y[i].sum(), degs, x_oh_pinv[0], x_oh_pinv[1],
                    x_oh_pinv[2], iters, overfit, False)

                end = time.time()

                w[m_x_cumsum[i]:m_x_cumsum[i + 1], :m_x_cumsum[
                    i]] = wi[:, :m_x_cumsum[i]]
                w[m_x_cumsum[i]:m_x_cumsum[i + 1], m_x_cumsum[
                    i + 1]:] = wi[:, m_x_cumsum[i]:]

                d.append(di[0][1:it[0]])

        w = {
            deg: w[:, mm_x_cumsum[i]:mm_x_cumsum[i + 1]]
            for i, deg in enumerate(degs)
        }

        self.impute = impute

        self.x_int = x_int
        self.cat_x = cat_x
        self.m_x = m_x
        self.n_x = n_x

        self.cat_x_inv = cat_x_inv
        self.cat_y_inv = cat_y_inv
        self.m_x_cumsum = m_x_cumsum
        self.m_y_cumsum = m_y_cumsum

        self.y_int = y_int
        self.cat_y = cat_y
        self.m_y = m_y
        self.n_y = n_y

        self.d = d
        self.w = w

    def predict(self, x):

        cat_x = self.cat_x
        w = self.w
        degs = self.degs
        m_x = self.m_x
        n_y = self.n_y
        m_y_cumsum = self.m_y_cumsum
        cat_y_inv = self.cat_y_inv

        x = np.array(x)
        if x.ndim == 1:
            x = x[:, np.newaxis]

        x_int = np.empty(x.shape, dtype=int)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x_int[i, j] = cat_x[i][x[i, j]]

        x_oh, idx_x = one_hot(x_int, m_x, degs)
        x_oh = x_oh.toarray()

        w = np.hstack(w.values())

        p = np.exp(w.dot(x_oh))
        p = np.split(p, m_y_cumsum[1:-1], axis=0)
        for i in range(n_y):
            p[i] /= p[i].sum(0)

        y_int = np.array([pi.argmax(axis=0) for pi in p])

        j = np.arange(y_int.shape[1])
        p = np.array([pi[yi, j] for yi, pi in zip(y_int, p)])

        y = np.empty(y_int.shape, dtype=x.dtype)
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                y[i, j] = cat_y_inv[i][y_int[i, j]]

        y = y.squeeze()
        p = p.squeeze()

        return y, p


def svd_pinv(x):

    x_rank = np.linalg.matrix_rank(x.todense())

    x_svd = svds(x, k=min(x_rank, min(x.shape) - 1))
    # x_oh_svd = svds(x_oh, k=x_oh_rank)

    sv_pinv = x_svd[1]
    zero_sv = np.isclose(sv_pinv, 0)
    sv_pinv[~zero_sv] = 1.0 / sv_pinv[~zero_sv]
    sv_pinv[zero_sv] = 0.0
    x_pinv = [x_svd[2].T, sv_pinv, x_svd[0].T]

    return x_pinv

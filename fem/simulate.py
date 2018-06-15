import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
import combinatorics
from fortran_module import fortran_module


def model_parameters(n, m, degs=[1], dist=None, dist_par=None):

    try:
        len(m)
    except:
        m = np.repeat(m, n)

    m_sum = m.sum()
    m_cumsum = np.insert(m.cumsum(), 0, 0)

    degs = np.array(degs)
    max_deg = degs.max()

    if (dist is None) or (dist_par is None):
        dist = np.random.normal
        dist_par = (0.0, 1.0 / np.sqrt(m.sum()))

    idx_by_deg = [combinatorics.multiindices(n, deg) for deg in degs]
    mi = [np.array([np.prod(m[i]) for i in idx]) for idx in idx_by_deg]
    mi_sum = [mii.sum() for mii in mi]
    mi_cumsum = [np.insert(mii.cumsum(), 0, 0) for mii in mi]

    par = {
        deg: dist(*dist_par, size=(m_sum, mi_sum[i]))
        for i, deg in enumerate(degs)
    }
    for (i, deg) in enumerate(degs):
        for (m1, m2) in zip(m_cumsum[:-1], m_cumsum[1:]):
            par[deg][m1:m2] -= par[deg][m1:m2].mean(0)
        for (m1, m2) in zip(mi_cumsum[i][:-1], mi_cumsum[i][1:]):
            par[deg][:, m1:m2] -= par[deg][:, m1:m2].mean(1)[:, np.newaxis]

    return par

def mutations(par, n, m, l=None, o=1.0):

    try:
        len(m)
    except:
        m = np.repeat(m, n)

    degs = np.sort(par.keys())

    par = np.hstack([par[deg] for deg in degs])

    if l is None:
        l = int(o * np.prod(par.shape))

    return fortran_module.simulate_mutations(par, m, l, degs)

def time_series(par, n, m, l=None, o=1.0):

    try:
        len(m)
    except:
        m = np.repeat(m, n)

    degs = np.sort(par.keys())

    par = np.hstack([par[deg] for deg in degs])

    if l is None:
        l = int(o * np.prod(par.shape))

    return fortran_module.simulate_time_series(par, m, l, degs)

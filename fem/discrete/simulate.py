import numpy as np
import combinatorics
from .. import fortran_module


def model_parameters(n, m, degs=[1], dist=None, dist_par=None):
    """Draw random model parameters

    Args:
        n (int):
        m (int):
        degs (list):
        dist (callable)
        dist_par (tuple):

    Returns:
        dict: keys `degs`

    """

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

    w = {
        deg: dist(*dist_par, size=(m_sum, mi_sum[i]))
        for i, deg in enumerate(degs)
    }
    for (i, deg) in enumerate(degs):
        for (m1, m2) in zip(m_cumsum[:-1], m_cumsum[1:]):
            w[deg][m1:m2] -= w[deg][m1:m2].mean(0)
        for (m1, m2) in zip(mi_cumsum[i][:-1], mi_cumsum[i][1:]):
            w[deg][:, m1:m2] -= w[deg][:, m1:m2].mean(1)[:, np.newaxis]

    return w


def time_series(w, n, m, l=None, o=1.0):
    """Simulate discrete time series data

    Args:
        w (dict):
        n (int):
        m (int):
        l (int):
        o (float)

    Returns:
        ndarray: time series data

    """

    try:
        len(m)
    except:
        m = np.repeat(m, n)

    degs = np.sort(w.keys())

    w = np.hstack([w[deg] for deg in degs])

    if l is None:
        l = int(o * np.prod(w.shape))

    return fortran_module.fortran_module.simulate_time_series(w, m, l, degs)


def mutations(w, n, m, l=None, o=1.0):

    try:
        len(m)
    except:
        m = np.repeat(m, n)

    degs = np.sort(w.keys())

    w = np.hstack([w[deg] for deg in degs])

    if l is None:
        l = int(o * np.prod(w.shape))

    return fortran_module.fortran_module.simulate_mutations(w, m, l, degs)

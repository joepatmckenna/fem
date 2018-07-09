import numpy as np


def binomial_coefficients(n, k):
    """Return a list of binomial coefficients.

    Args:
        n (int): the first argument to the *choose* operation
        k (int): the maximum second argument to the *choose* operation

    Returns:
        ndarray: a list of binomial coefficients

    Examples:

        Compute the binomial coefficients :math:`{4\choose 0}`, :math:`{4\choose 1}`, and :math:`{4\choose 2}`

        >>> from fem.discrete.combinatorics import binomial_coefficients
        >>> binomial_coefficients(4, 2)
        array([1, 4, 6])
    """
    seq = np.empty(k + 1, dtype=int)
    seq[0] = 1
    for i in range(1, k + 1):
        seq[i] = seq[i - 1] * (n - i + 1) / i
    return seq


def mixed_radix_to_base_10(x, b):
    """Convert the `mixed radix`_ integer with digits `x` and bases `b` to base 10.

    Args:
        x (list): a list of digits ordered by increasing place values
        b (list): a list of bases corresponding to the digits

    Examples:
        Generally, the base 10 representation of the mixed radix number :math:`x_n\ldots x_1` where :math:`x_i` is a digit in place value :math:`i` with base :math:`b_i` is

        .. math::

            \sum_{i=1}^nx_i\prod_{j=i+1}^nb_j = x_n + b_nx_{n-1} + b_nb_{n-1}x_{n-2} + \cdots + b_n\cdots b_2x_1

        Convert 111 with bases :math:`(b_1,b_2,b_3)=(2,3,4)` to base 10:

        >>> from fem.discrete.combinatorics import mixed_radix_to_base_10
        >>> mixed_radix_to_base_10([1,1,1], [2,3,4])
        17

    .. _mixed radix:
        https://en.wikipedia.org/wiki/Mixed_radix

    """
    res = x[0]
    for i in range(1, len(x)):
        res *= b[i]
        res += x[i]
    return res


def multiindices(n, k):
    """Return an ordered list of distinct tuples of the first :math:`n` nonnegative integers.

    Args:
        n (int): The least integer not included in the list of distinct tuples
        k (int): The length of each tuple

    Returns:
        ndarray: a list of distinct tupes of shape :math:`{n\choose k}` by :math:`k`

    Examples:

        >>> from fem.discrete.combinatorics import multiindices
        >>> multiindices(4, 2)
        array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])

    """

    if k > n:
        return []

    p = binomial_coefficients(n, k)[-1]

    var = np.empty((p, k), dtype=int)
    v = np.arange(k)
    var[0] = v.copy()

    for i in range(1, p):

        idx = k - 1
        v[idx] += 1
        while v[idx] + k - idx > n:
            idx -= 1
            v[idx] += 1
        for idx in range(idx + 1, k):
            v[idx] = v[idx - 1] + 1

        var[i] = v.copy()

    return var

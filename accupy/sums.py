import math
from typing import Tuple

import _accupy
import numpy as np
from numpy.typing import ArrayLike


def knuth_sum(a: float, b: float) -> Tuple[float, float]:
    """Error-free transformation of the sum of two floating point numbers according to

    D.E. Knuth.
    The Art of Computer Programming: Seminumerical Algorithms, volume 2.
    Addison Wesley, Reading, Massachusetts, second edition, 1981.

    The underlying problem is that the exact sum a+b of two floating point number a and
    b is not necessarily a floating point number; for example if you add a very large
    and a very small number. It is however known that the difference between the best
    floating point approximation of a+b and the exact a+b is again a floating point
    number. This routine returns the sum and the error.

    Algorithm 3.1 in <https://doi.org/10.1137/030601818>.
    """
    x = a + b
    z = x - a
    y = (a - (x - z)) + (b - z)
    return x, y


def decker_sum(a: float, b: float) -> Tuple[float, float]:
    """Computationally equivalent to knuth_sum, but formally a bit cheaper.
    Only works for floats though (and not arrays), and the branch make it in
    fact less favorable in terms of actual speed.
    """
    x = a + b
    y = b - (x - a) if abs(a) > abs(b) else a - (x - b)
    return x, y


def distill(p: ArrayLike, K: int) -> np.ndarray:
    """Algorithm 4.3. Error-free vector transformation for summation.

    The vector p is transformed without changing the sum, and p_n is replaced
    by float(sum(p)). Kahan [21] calls this a 'distillation algorithm.'
    """
    p = np.asarray(p)

    q = p.reshape(p.shape[0], -1)
    for _ in range(K):
        _accupy.distill(q)
    return q.reshape(p.shape)


def ksum(p: ArrayLike, K: int = 2) -> float:
    """From

    T. Ogita, S.M. Rump, and S. Oishi.
    Accurate Sum and Dot Product,
    SIAM J. Sci. Comput., 26(6), 1955–1988 (34 pages).
    <https://doi.org/10.1137/030601818>.

    Algorithm 4.8. Summation as in K-fold precision by (K−1)-fold error-free
    vector transformation.
    """
    # Don't override the input data.
    p = np.asarray(p)
    q = p.copy()
    distill(q, K - 1)
    return np.sum(q[:-1], axis=0) + q[-1]


_math_fsum_vec = np.vectorize(math.fsum, signature="(m)->()")


def fsum(p: ArrayLike) -> float:
    p = np.asarray(p)
    return _math_fsum_vec(p.T).T


def kahan_sum(p: ArrayLike) -> float:
    """Kahan summation
    <https://en.wikipedia.org/wiki/Kahan_summation_algorithm>.
    """
    p = np.asarray(p)
    q = p.reshape(p.shape[0], -1)
    s = _accupy.kahan(q)
    return s.reshape(p.shape[1:])

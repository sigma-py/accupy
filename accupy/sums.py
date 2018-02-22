# -*- coding: utf-8 -*-
#
import math

import numpy

import _accupy


def knuth_sum(a, b):
    '''Error-free transformation of the sum of two floating point numbers
    according to

    D.E. Knuth.
    The Art of Computer Programming: Seminumerical Algorithms, volume 2.
    Addison Wesley, Reading, Massachusetts, second edition, 1981.

    The underlying problem is that the exact sum a+b of two floating point
    number a and b is not necessarily a floating point number; for example if
    you add a very large and a very small number. It is however known that the
    difference between the best floating point approximation of a+b and the
    exact a+b is again a floating point number. This routine returns the sum
    and the error.

    Algorithm 3.1 in <https://doi.org/10.1137/030601818>.
    '''
    x = a + b
    z = x - a
    y = (a - (x-z)) + (b-z)
    return x, y


def decker_sum(a, b):
    '''Computationally equivalent to knuth_sum, but formally a bit cheaper.
    Only works for floats though (and not arrays), and the branch make it in
    fact less favorable in terms of actual speed.
    '''
    x = a + b
    if abs(a) > abs(b):
        y = b - (x-a)
    else:
        y = a - (x-b)
    return x, y


# def distill_python(p):
#     '''Algorithm 4.3. Error-free vector transformation for summation.
#
#     The vector p is transformed without changing the sum, and p_n is replaced
#     by float(sum(p)). Kahan [21] calls this a "distillation algorithm."
#     '''
#     for i in range(1, len(p)):
#         p[i], p[i-1] = knuth_sum(p[i], p[i-1])
#     return p


def distill(p):
    '''Algorithm 4.3. Error-free vector transformation for summation.

    The vector p is transformed without changing the sum, and p_n is replaced
    by float(sum(p)). Kahan [21] calls this a "distillation algorithm."
    '''
    # Append newaxis to account for len(p.shape)==1
    p = p[..., numpy.newaxis]
    q = p.reshape(p.shape[0], numpy.prod(p.shape[1:]))
    _accupy.distill(q)
    return q.reshape(p.shape[:-1])


def ksum(p, K=2):
    '''From

    T. Ogita, S.M. Rump, and S. Oishi.
    Accurate Sum and Dot Product,
    SIAM J. Sci. Comput., 26(6), 1955–1988 (34 pages).
    <https://doi.org/10.1137/030601818>.

    Algorithm 4.8. Summation as in K-fold precision by (K−1)-fold error-free
    vector transformation.
    '''
    # Don't override the input data.
    q = p.copy()
    for _ in range(1, K):
        distill(q)
    return sum(q[:-1]) + q[-1]


_math_fsum_vec = numpy.vectorize(math.fsum, signature="(m)->()")


def fsum(p):
    return _math_fsum_vec(p.T).T


_kahan_vec = numpy.vectorize(_accupy.kahan, signature="(m)->()")

def kahan_sum(p):
    '''Kahan summation of the numpy array `a` along axis `axis`.
    '''
    # See <https://en.wikipedia.org/wiki/Kahan_summation_algorithm> for
    # details.
    return _kahan_vec(p.T).T

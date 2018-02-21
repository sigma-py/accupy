# -*- coding: utf-8 -*-
#
import pyfma


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


def oro_sum(p):
    '''From

    T. Ogita, S.M. Rump, and S. Oishi.
    Accurate Sum and Dot Product,
    SIAM J. Sci. Comput., 26(6), 1955–1988 (34 pages).
    <https://doi.org/10.1137/030601818>.
    '''
    for i in range(1, len(p)):
        p[i], p[i-1] = knuth_sum(p[i], p[i-1])
    return sum(p[:-1]) + p[-1]


def generate_ill_conditioned_sum(n, c):
    # From <https://doi.org/10.1137/030601818>:
    # Ill-conditioned sums of length 2n are generated from dot products of
    # length n using Algorithm 3.3 (TwoProduct) and randomly permuting the
    # summands.
    from mpmath import mp
    import numpy
    x, y, d, C = generate_ill_conditioned_dot_product(n, c)

    res = [list(prod_fma(xx, yy)) for xx, yy in zip(x, y)]

    out = numpy.random.permutation(
        [item for sublist in res for item in sublist]
        )

    def sum_exact(p):
        mp.dps = 100
        # convert to list first, see
        # <https://github.com/fredrik-johansson/mpmath/pull/385>
        return float(mp.fsum(p))

    return out, sum_exact(out)


def generate_ill_conditioned_dot_product(n, c):
    '''n ... length of vector
    c ... target condition number
    '''
    import math
    from mpmath import mp
    import numpy

    numpy.random.seed(0)

    # Algorithm 6.1 from
    #
    # ACCURATE SUM AND DOT PRODUCT,
    # TAKESHI OGITA, SIEGFRIED M. RUMP, AND SHIN'ICHI OISHI.
    assert n >= 6
    n2 = round(n / 2)
    x = numpy.zeros(n)
    y = numpy.zeros(n)

    b = math.log2(c)
    # vector of exponents between 0 and b/2:
    e = numpy.rint(numpy.random.rand(n2) * b/2).astype(int)
    # make sure exponents b/2 and 0 actually occur in e
    # vectors x,y
    e[0] = round(b/2) + 1
    e[-1] = 0

    # generate first half of vectors x, y
    rx, ry = numpy.random.rand(2, n2)
    x[:n2] = (2*rx - 1) * 2**e
    y[:n2] = (2*ry - 1) * 2**e

    def dot_exact(x, y):
        mp.dps = 100
        # convert to list first, see
        # <https://github.com/fredrik-johansson/mpmath/pull/385>
        return float(mp.fdot(x.tolist(), y.tolist()))

    # for i=n2+1:n and v=1:i,
    #     generate x_i, y_i such that (*) x(v)’*y(v) ~ 2^e(i-n2)
    # generate exponents for second half
    e = numpy.rint(numpy.linspace(b/2, 0, n-n2)).astype(int)
    rx, ry = numpy.random.rand(2, n2)
    for i in range(n2, n):
        # x_i random with generated exponent
        x[i] = (2*rx[i-n2] - 1) * 2**e[i-n2]
        # y_i according to (*)
        y[i] = (
            (2*ry[i-n2] - 1) * 2**e[i-n2]
            - dot_exact(x[:i+1], y[:i+1])
            ) / x[i]

    x, y = numpy.random.permutation((x, y))
    # the true dot product rounded to nearest floating point
    d = dot_exact(x, y)
    # the actual condition number
    C = 2 * dot_exact(abs(x), abs(y)) / abs(d)

    return x, y, d, C


def split(a):
    '''Error-free splitting of a floating-point number into two parts.
    '''
    s = 27
    factor = 2*s + 1
    c = factor * a
    x = c - (c-a)
    y = a - x
    return x, y


def prod(a, b):
    '''Error-free transformation of the product of two floating-point numbers.
    '''
    x = a * b
    a1, a2 = split(a)
    b1, b2 = split(b)
    y = a2*b2 - (((x - a1*b1) - a2*b1) - a1*b2)
    return x, y


def prod_fma(a, b):
    '''Error-free transformation of a product using Fused-Multiply-and-Add.
    '''
    x = a * b
    y = pyfma.fma(a, b, -x)
    return x, y

import math

import numpy
import pyfma
from mpmath import mp


def generate_ill_conditioned_sum(n, c, dps=100):
    # From <https://doi.org/10.1137/030601818>:
    # Ill-conditioned sums of length 2n are generated from dot products of
    # length n using Algorithm 3.3 (TwoProduct) and randomly permuting the
    # summands.
    x, y, _, C = generate_ill_conditioned_dot_product(n, c, dps)

    prod = x * y
    err = pyfma.fma(x, y, -prod)
    res = numpy.array([prod, err])

    out = numpy.random.permutation(res.flatten())

    def sum_exact(p):
        mp.dps = dps
        return mp.fsum(p)

    exact = sum_exact(out)
    # cond = sum_exact(numpy.abs(out)) / abs(exact)
    cond = C / 2

    return out, exact, cond


def generate_ill_conditioned_dot_product(n, c, dps=100):
    """n ... length of vector
    c ... target condition number
    """
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
    e = numpy.rint(numpy.random.rand(n2) * b / 2).astype(int)
    # make sure exponents b/2 and 0 actually occur in e
    # vectors x,y
    e[0] = round(b / 2) + 1
    e[-1] = 0

    # generate first half of vectors x, y
    rx, ry = numpy.random.rand(2, n2)
    x[:n2] = (2 * rx - 1) * 2 ** e
    y[:n2] = (2 * ry - 1) * 2 ** e

    def dot_exact(x, y):
        mp.dps = dps
        # convert to list first, see
        # <https://github.com/fredrik-johansson/mpmath/pull/385>
        return mp.fdot(x.tolist(), y.tolist())

    # for i=n2+1:n and v=1:i,
    #     generate x_i, y_i such that (*) x(v)’*y(v) ~ 2^e(i-n2)
    # generate exponents for second half
    e = numpy.rint(numpy.linspace(b / 2, 0, n - n2)).astype(int)
    rx, ry = numpy.random.rand(2, n2)
    for i in range(n2, n):
        # x_i random with generated exponent
        x[i] = (2 * rx[i - n2] - 1) * 2 ** e[i - n2]
        # y_i according to (*)
        y[i] = (
            (2 * ry[i - n2] - 1) * 2 ** e[i - n2] - dot_exact(x[: i + 1], y[: i + 1])
        ) / x[i]

    x, y = numpy.random.permutation((x, y))
    # the true dot product rounded to nearest floating point
    d = dot_exact(x, y)
    # the actual condition number
    C = 2 * dot_exact(abs(x), abs(y)) / abs(d)

    return x, y, d, C

# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy

import accupy


def test_ill_conditioned_sum():
    import math
    p, exact = accupy.generate_ill_conditioned_sum(100, 1.0e38)
    print()
    print(exact)
    print()
    print(numpy.sum(p))
    s = 0.0
    for item in p:
        s += item
    print(s)
    print(sum(p))
    print()
    print(math.fsum(p))
    print()
    print(accupy.kahan_sum(p))
    print(accupy.fsum(p, K=1))
    print(accupy.fsum(p, K=2))
    print(accupy.fsum(p, K=3))
    print(accupy.fsum(p, K=4))
    return


def test_knuth_sum():
    a16 = numpy.float16(1.0e+1)
    b16 = numpy.float16(1.0e-1)

    x16, y16 = accupy.knuth_sum(a16, b16)

    xy = numpy.float64(x16) + numpy.float64(y16)
    ab = numpy.float64(a16) + numpy.float64(b16)

    assert abs(xy - ab) < 1.0e-15*ab
    return


def test_decker_sum():
    a16 = numpy.float16(1.0e+1)
    b16 = numpy.float16(1.0e-1)

    x16, y16 = accupy.decker_sum(a16, b16)

    xy = numpy.float64(x16) + numpy.float64(y16)
    ab = numpy.float64(a16) + numpy.float64(b16)

    assert abs(xy - ab) < 1.0e-15*ab
    return


def test_sum():
    # Test with geometric sum
    n = 10000
    # p = numpy.float16(1.0) / numpy.arange(1, n)
    p = numpy.random.rand(n) / n
    s = accupy.fsum(p)
    print(s)
    s2 = 0.0
    for r in p:
        s2 += r
    print(s2)
    return


if __name__ == '__main__':
    test_ill_conditioned_sum()

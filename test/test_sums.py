# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy

import accupy


def test_ill_conditioned_sum():
    p, exact = accupy.generate_ill_conditioned_sum(10, 1.0e20)
    print(p)
    print(exact)
    print(numpy.sum(p))
    print(sum(p))
    s = 0.0
    for item in p:
        s += item
    print(s)
    print(accupy.oro_sum(p))
    print(accupy.kahan_sum(p))
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
    s = accupy.oro_sum(p)
    print(s)
    s2 = 0.0
    for r in p:
        s2 += r
    print(s2)
    return


if __name__ == '__main__':
    test_ill_conditioned_sum()

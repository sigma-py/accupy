# -*- coding: utf-8 -*-
#
from __future__ import division

import numpy

import accupy

numpy.random.seed(0)


def test_prod():
    a = 2.0
    b = 1.0e10
    x, y = accupy.prod2(a, b)
    assert x == 2.0e10
    assert y == 0.0
    return


def test_dot1():
    x, y, d, C = accupy.generate_ill_conditioned_dot_product(10, 1.0e20)
    print(d)
    print()
    print(numpy.dot(x, y))
    print(accupy.dot1(x, y))
    print(accupy.dot2(x, y))
    print(accupy.dotK(x, y))
    return


if __name__ == '__main__':
    test_dot1()

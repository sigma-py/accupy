# -*- coding: utf-8 -*-
#
from __future__ import division

import accupy


def test_prod():
    a = 2.0
    b = 1.0e10
    x, y = accupy.prod(a, b)
    assert x == 2.0e10
    assert y == 0.0
    return


if __name__ == '__main__':
    test_prod()

# -*- coding: utf-8 -*-
#
import numpy

import accupy


def test_knuth_sum():
    a16 = numpy.float16(1.0e+1)
    b16 = numpy.float16(1.0e-1)

    x16, y16 = accupy.knuth_sum(a16, b16)

    xy = numpy.float64(x16) + numpy.float64(y16)
    ab = numpy.float64(a16) + numpy.float64(b16)

    assert abs(xy - ab) < 1.0e-15*ab
    return


if __name__ == '__main__':
    test_knuth_sum()

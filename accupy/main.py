# -*- coding: utf-8 -*-
#


def knuth_sum(a, b):
    '''Error-free transformation of the sum of two floating point numbers
    according to

    D.E. Knuth.
    The Art of Computer Programming: Seminumerical Algorithms, volume 2.
    Addison Wesley, Reading, Massachusetts, second edition, 1981.
    '''
    x = a + b
    z = x - a
    y = (a - (x-z)) + (b-z)
    return x, y

# -*- coding: utf-8 -*-
#
import pyfma

import _accupy

from .sums import ksum, knuth_sum


def split(a):
    '''Error-free splitting of a floating-point number into two parts.

    Algorithm 3.2 in <https://doi.org/10.1137/030601818>.
    '''
    s = 27
    factor = 2*s + 1
    c = factor * a
    x = c - (c-a)
    y = a - x
    return x, y


def prod2_split(a, b):
    '''Error-free transformation of the product of two floating-point numbers.

    Algorithm 3.3 in <https://doi.org/10.1137/030601818>.
    '''
    x = a * b
    a1, a2 = split(a)
    b1, b2 = split(b)
    y = a2*b2 - (((x - a1*b1) - a2*b1) - a1*b2)
    return x, y


def prod2_fma(a, b):
    '''Error-free transformation of a product using Fused-Multiply-and-Add.

    Algorithm 3.5 in <https://doi.org/10.1137/030601818>.
    '''
    x = a * b
    y = pyfma.fma(a, b, -x)
    return x, y


def dot2(x, y, prod2=prod2_fma):
    '''Algorithm 5.3. Dot product in twice the working precision.
    in <https://doi.org/10.1137/030601818>.
    '''
    p, s = prod2(x[0], y[0])
    n = len(x)
    for k in range(1, n):
        h, r = prod2(x[k], y[k])
        p, q = knuth_sum(p, h)
        s += q+r
    return p + s


def kdot(x, y, K=2, prod2=prod2_fma):
    '''Algorithm 5.10. Dot product algorithm in K-fold working precision,
    K >= 3.
    '''
    r = _accupy.kdot_helper(x, y)
    return ksum(r, K-1)

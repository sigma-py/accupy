# -*- coding: utf-8 -*-
#
import numpy
import pyfma

from .sums import fsum, knuth_sum


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


def prod2(a, b):
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


# def dot1(x, y, K=1, prod2=prod2_fma):
#     '''Algorithm 5.1. A first dot product algorithm.
#     in <https://doi.org/10.1137/030601818>.
#     '''
#     [r0, r1] = prod2(x, y)
#     r = numpy.concatenate([r0, r1])
#     return fsum(r, K)


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


def dotK(x, y, K=2, prod2=prod2_fma):
    '''Algorithm 5.10. Dot product algorithm in K-fold working precision,
    K >= 3.
    '''
    r0 = numpy.empty(x.shape)
    r1 = numpy.empty(x.shape)
    p, r0[0] = prod2(x[0], y[0])
    n = len(x)
    for k in range(1, n):
        h, r0[k] = prod2(x[k], y[k])
        p, r1[k-1] = knuth_sum(p, h)
    r1[-1] = p
    r = numpy.concatenate([r0, r1])
    return fsum(r, K-1)

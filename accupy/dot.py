# -*- coding: utf-8 -*-
#
import pyfma

from .sums import oro_sum


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


def dot1(x, y, K=1, prod=prod2_fma):
    '''Algorithm 5.1. A first dot product algorithm.
    in <https://doi.org/10.1137/030601818>.
    '''
    [r0, r1] = prod2(x, y)
    print(r0, r1)
    exit(1)
    return oro_sum(r, K)

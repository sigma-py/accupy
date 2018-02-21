# -*- coding: utf-8 -*-
#
import pyfma


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


def prod(a, b):
    '''Error-free transformation of the product of two floating-point numbers.

    Algorithm 3.3 in <https://doi.org/10.1137/030601818>.
    '''
    x = a * b
    a1, a2 = split(a)
    b1, b2 = split(b)
    y = a2*b2 - (((x - a1*b1) - a2*b1) - a1*b2)
    return x, y


def prod_fma(a, b):
    '''Error-free transformation of a product using Fused-Multiply-and-Add.

    Algorithm 3.5 in <https://doi.org/10.1137/030601818>.
    '''
    x = a * b
    y = pyfma.fma(a, b, -x)
    return x, y

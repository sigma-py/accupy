# -*- coding: utf-8 -*-
#
from __future__ import division

import matplotlib.pyplot as plt
import numpy
import pytest

import accupy

numpy.random.seed(0)


@pytest.mark.parametrize('cond', [1.0, 1.0e10, 1.0e15])
def test_ill_conditioned_sum(cond):
    p, ref = accupy.generate_ill_conditioned_sum(100, cond)
    assert abs(accupy.fsum(p, K=2) - ref) < 1.0e-15 * abs(ref)
    return


def test_accuracy_comparison():
    kernels = [
        numpy.sum,
        lambda p: accupy.ksum(p, K=2),
        lambda p: accupy.ksum(p, K=3),
        accupy.fsum,
        ]
    labels = [
        'numpy.sum',
        'ksum[2]',
        'ksum[3]',
        'fsum',
        ]
    x = [10**k for k in range(0, 37, 3)]
    data = numpy.empty((len(x), len(kernels)))
    condition_numbers = numpy.empty(len(x))
    for k, target_cond in enumerate(x):
        p, ref, C = accupy.generate_ill_conditioned_sum(1000, target_cond)
        condition_numbers[k] = C
        data[k] = [abs(kernel(p) - ref) / abs(ref) for kernel in kernels]

    for label, d in zip(labels, data.T):
        plt.loglog(condition_numbers, d, label=label)

    plt.legend()
    plt.grid()
    plt.ylim(5.0e-18, 1.0)
    plt.xlabel('condition number')
    plt.ylabel('error')

    # plt.show()
    plt.savefig('accuracy-sums.png', transparent=True)
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
    test_accuracy_comparison()

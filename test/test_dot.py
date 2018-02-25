# -*- coding: utf-8 -*-
#
from __future__ import division

import matplotlib.pyplot as plt
import numpy
import pytest
import perfplot

import accupy

numpy.random.seed(0)


def test_prod():
    a = 2.0
    b = 1.0e10
    x, y = accupy.prod2_split(a, b)
    assert x == 2.0e10
    assert y == 0.0
    return


@pytest.mark.parametrize('target_cond', [
    [10**k for k in range(5)]
    ])
def test_accuracy_comparison_illcond(target_cond):
    kernels = [
        numpy.dot,
        lambda x, y: accupy.kdot(x, y, K=2),
        lambda x, y: accupy.kdot(x, y, K=3),
        accupy.fdot,
        ]
    labels = [
        'numpy.dot',
        'accupy.kdot[2]',
        'accupy.kdot[3]',
        'accupy.fdot',
        ]
    data = numpy.empty((len(target_cond), len(kernels)))
    condition_numbers = numpy.empty(len(target_cond))
    for k, target_cond in enumerate(target_cond):
        x, y, ref, C = \
            accupy.generate_ill_conditioned_dot_product(1000, target_cond)
        condition_numbers[k] = C
        data[k] = [abs(kernel(x, y) - ref) / abs(ref) for kernel in kernels]

    for label, d in zip(labels, data.T):
        plt.loglog(condition_numbers, d, label=label)

    plt.legend()
    plt.grid()
    plt.ylim(5.0e-18, 1.0)
    plt.xlabel('condition number')
    plt.ylabel('relative error')
    plt.gca().set_aspect(1.3)

    # plt.show()
    plt.savefig('accuracy-dot.png', transparent=True)
    return


@pytest.mark.parametrize('n_range', [
    [2**k for k in range(5)]
    ])
def test_speed_comparison1(n_range):
    perfplot.save(
        'speed-comparison1.png',
        setup=lambda n: numpy.random.rand(2, n, 100),
        kernels=[
            lambda xy: numpy.dot(*xy),
            lambda xy: accupy.kdot(*xy, K=2),
            lambda xy: accupy.kdot(*xy, K=3),
            lambda xy: accupy.fdot(*xy),
            ],
        labels=[
            'numpy.dot',
            'accupy.kdot[2]',
            'accupy.kdot[3]',
            'accupy.fdot',
            ],
        n_range=n_range,
        title='dot(random(n, 100), random(n, 100))',
        xlabel='n',
        logx=True,
        logy=True,
        )
    return


if __name__ == '__main__':
    # test_accuracy_comparison_illcond([10**k for k in range(0, 37, 3)])
    test_speed_comparison1(n_range=[2**k for k in range(15)])

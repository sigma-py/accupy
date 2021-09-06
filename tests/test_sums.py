import dufte
import matplotlib.pyplot as plt
import numpy as np
import perfplot
import pytest

import accupy


def test_cond():
    cond = accupy.cond([1.0, 1.0e-16, -1.0])
    ref = 2.0e16
    assert abs(cond - ref) < 1.0e-13 * abs(ref)


@pytest.mark.parametrize("cond", [1.0, 1.0e15])
def test_ksum2(cond):
    p, ref, _ = accupy.generate_ill_conditioned_sum(100, cond)
    assert abs(accupy.ksum(p, K=2) - ref) < 1.0e-15 * abs(ref)


@pytest.mark.parametrize("cond", [1.0, 1.0e15, 1.0e30])
def test_ksum3(cond):
    p, ref, _ = accupy.generate_ill_conditioned_sum(100, cond)
    assert abs(accupy.ksum(p, K=3) - ref) < 1.0e-15 * abs(ref)


@pytest.mark.parametrize("cond", [1.0, 1.0e15, 1.0e30, 1.0e35])
def test_fsum(cond):
    p, ref, _ = accupy.generate_ill_conditioned_sum(100, cond)
    assert abs(accupy.fsum(p) - ref) < 1.0e-15 * abs(ref)


def test_accuracy_comparison_illcond(target_conds=None):
    plt.style.use(dufte.style)

    if target_conds is None:
        target_conds = [10 ** k for k in range(1, 2)]

    kernels = [
        sum,
        np.sum,
        accupy.kahan_sum,
        lambda p: accupy.ksum(p, K=2),
        lambda p: accupy.ksum(p, K=3),
        accupy.fsum,
    ]
    labels = [
        "sum",
        "np.sum",
        "accupy.kahan_sum",
        "accupy.ksum[2]",
        "accupy.ksum[3]",
        "accupy.fsum",
    ]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(labels)]

    data = np.empty((len(target_conds), len(kernels)))
    condition_numbers = np.empty(len(target_conds))
    np.random.seed(0)
    for k, target_cond in enumerate(target_conds):
        p, ref, C = accupy.generate_ill_conditioned_sum(1000, target_cond)
        condition_numbers[k] = C
        data[k] = [abs(kernel(p) - ref) / abs(ref) for kernel in kernels]

    # sort
    s = np.argsort(condition_numbers)
    condition_numbers = condition_numbers[s]
    data = data[s]

    for label, color, d in zip(labels, colors, data.T):
        plt.loglog(condition_numbers, d, label=label, color=color)

    dufte.legend()
    plt.xlabel("condition number")
    dufte.ylabel("relative error")
    # plt.gca().set_aspect(1.3)


def test_speed_comparison1(n_range=None):
    plt.style.use(dufte.style)

    if n_range is None:
        n_range = [2 ** k for k in range(2)]

    np.random.seed(0)
    perfplot.plot(
        setup=lambda n: np.random.rand(n, 100),
        kernels=[
            sum,
            lambda p: np.sum(p, axis=0),
            accupy.kahan_sum,
            lambda p: accupy.ksum(p, K=2),
            lambda p: accupy.ksum(p, K=3),
            accupy.fsum,
        ],
        labels=[
            "sum",
            "np.sum",
            "accupy.kahan_sum",
            "accupy.ksum[2]",
            "accupy.ksum[3]",
            "accupy.fsum",
        ],
        n_range=n_range,
        xlabel="n",
    )
    plt.title("Sum(random(n, 100))")


def test_speed_comparison2(n_range=None):
    plt.style.use(dufte.style)

    if n_range is None:
        n_range = [2 ** k for k in range(2)]

    np.random.seed(0)
    perfplot.plot(
        setup=lambda n: np.random.rand(100, n),
        kernels=[
            sum,
            lambda p: np.sum(p, axis=0),
            accupy.kahan_sum,
            lambda p: accupy.ksum(p, K=2),
            lambda p: accupy.ksum(p, K=3),
            accupy.fsum,
        ],
        labels=[
            "sum",
            "np.sum",
            "accupy.kahan_sum",
            "accupy.ksum[2]",
            "accupy.ksum[3]",
            "accupy.fsum",
        ],
        n_range=n_range,
        xlabel="n",
    )
    plt.title("Sum(random(100, n))")


def test_knuth_sum():
    a16 = np.float16(1.0e1)
    b16 = np.float16(1.0e-1)
    x16, y16 = accupy.knuth_sum(a16, b16)
    xy = np.float64(x16) + np.float64(y16)
    ab = np.float64(a16) + np.float64(b16)
    assert abs(xy - ab) < 1.0e-15 * ab


def test_decker_sum():
    a16 = np.float16(1.0e1)
    b16 = np.float16(1.0e-1)
    x16, y16 = accupy.decker_sum(a16, b16)
    xy = np.float64(x16) + np.float64(y16)
    ab = np.float64(a16) + np.float64(b16)
    assert abs(xy - ab) < 1.0e-15 * ab


def test_discontiguous():
    x = np.random.rand(3, 10).T
    accupy.ksum(x.T)
    accupy.fsum(x.T)


if __name__ == "__main__":
    test_accuracy_comparison_illcond([10 ** k for k in range(0, 37, 1)])
    plt.savefig("accuracy-sum.svg", transparent=True, bbox_inches="tight")
    plt.close()

    test_speed_comparison1(n_range=[2 ** k for k in range(15)])
    plt.savefig("speed-comparison1.svg", transparent=True, bbox_inches="tight")
    plt.close()

    test_speed_comparison2(n_range=[2 ** k for k in range(15)])
    plt.savefig("speed-comparison2.svg", transparent=True, bbox_inches="tight")
    plt.close()

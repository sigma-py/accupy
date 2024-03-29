import dufte
import matplotlib.pyplot as plt
import numpy as np
import perfplot
import pytest

import accupy


def test_cond():
    cond = accupy.cond([np.pi, np.e], [23225 / 8544, -355 / 113])
    print(cond)
    ref = 4.852507317687677e7
    assert abs(cond - ref) < 1.0e-13 * abs(ref)


@pytest.mark.parametrize("cond", [1.0, 1.0e15])
def test_kdot2(cond):
    x, y, ref, _ = accupy.generate_ill_conditioned_dot_product(100, cond)
    assert abs(accupy.kdot(x, y, K=2) - ref) < 1.0e-13 * abs(ref)


@pytest.mark.parametrize("cond", [1.0, 1.0e15, 1.0e30])
def test_kdot3(cond):
    x, y, ref, _ = accupy.generate_ill_conditioned_dot_product(100, cond)
    assert abs(accupy.kdot(x, y, K=3) - ref) < 1.0e-13 * abs(ref)


@pytest.mark.parametrize("cond", [1.0, 1.0e15, 1.0e30, 1.0e38])
def test_fdot(cond):
    x, y, ref, _ = accupy.generate_ill_conditioned_dot_product(100, cond)
    assert abs(accupy.fdot(x, y) - ref) < 1.0e-13 * abs(ref)


def test_accuracy_comparison_illcond(target_cond=None):
    plt.style.use(dufte.style)

    if target_cond is None:
        target_cond = [10 ** k for k in range(2)]

    kernels = [
        np.dot,
        lambda x, y: accupy.kdot(x, y, K=2),
        lambda x, y: accupy.kdot(x, y, K=3),
        accupy.fdot,
    ]
    labels = ["np.dot", "accupy.kdot[2]", "accupy.kdot[3]", "accupy.fdot"]
    data = np.empty((len(target_cond), len(kernels)))
    condition_numbers = np.empty(len(target_cond))
    np.random.seed(0)
    for k, tc in enumerate(target_cond):
        x, y, ref, C = accupy.generate_ill_conditioned_dot_product(1000, tc)
        condition_numbers[k] = C
        data[k] = [abs(kernel(x, y) - ref) / abs(ref) for kernel in kernels]

    # sort
    s = np.argsort(condition_numbers)
    condition_numbers = condition_numbers[s]
    data = data[s]

    for label, d in zip(labels, data.T):
        plt.loglog(condition_numbers, d, label=label)

    dufte.legend()
    plt.xlabel("condition number")
    dufte.ylabel("relative error")


def test_speed_comparison1(n_range=None):
    plt.style.use(dufte.style)

    if n_range is None:
        n_range = [2 ** k for k in range(2)]

    np.random.seed(0)
    perfplot.plot(
        setup=lambda n: (np.random.rand(n, 100), np.random.rand(100, n)),
        kernels=[
            lambda xy: np.dot(*xy),
            lambda xy: accupy.kdot(*xy, K=2),
            lambda xy: accupy.kdot(*xy, K=3),
            lambda xy: accupy.fdot(*xy),
        ],
        labels=["np.dot", "accupy.kdot[2]", "accupy.kdot[3]", "accupy.fdot"],
        n_range=n_range,
        xlabel="n",
    )
    plt.title("dot(random(n, 100), random(100, n))")


def test_speed_comparison2(n_range=None):
    if n_range is None:
        n_range = [2 ** k for k in range(2)]

    np.random.seed(0)
    perfplot.plot(
        setup=lambda n: (np.random.rand(100, n), np.random.rand(n, 100)),
        kernels=[
            lambda xy: np.dot(*xy),
            lambda xy: accupy.kdot(*xy, K=2),
            lambda xy: accupy.kdot(*xy, K=3),
            lambda xy: accupy.fdot(*xy),
        ],
        labels=["np.dot", "accupy.kdot[2]", "accupy.kdot[3]", "accupy.fdot"],
        n_range=n_range,
        xlabel="n",
        logx=True,
        logy=True,
    )
    plt.title("dot(random(100, n), random(n, 100))")


def test_discontiguous():
    x = np.random.rand(3, 10)
    y = np.random.rand(3, 10)
    accupy.kdot(x.T, y)
    accupy.fdot(x.T, y)


if __name__ == "__main__":
    test_accuracy_comparison_illcond([10 ** k for k in range(0, 37, 1)])
    plt.savefig("accuracy-dot.svg", transparent=True, bbox_inches="tight")
    plt.close()

    test_speed_comparison1(n_range=[2 ** k for k in range(8)])
    plt.savefig("speed-comparison-dot1.svg", transparent=True, bbox_inches="tight")
    plt.close()

    test_speed_comparison2(n_range=[2 ** k for k in range(8)])
    plt.savefig("speed-comparison-dot2.svg", transparent=True, bbox_inches="tight")
    plt.close()

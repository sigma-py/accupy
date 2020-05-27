import matplotlib.pyplot as plt
import numpy
import pytest

import accupy
import perfplot


@pytest.mark.parametrize("cond", [1.0, 1.0e15])
def test_kdot2(cond):
    x, y, ref, _ = accupy.generate_ill_conditioned_dot_product(100, cond)
    assert abs(accupy.kdot(x, y, K=2) - ref) < 1.0e-15 * abs(ref)
    return


@pytest.mark.parametrize("cond", [1.0, 1.0e15, 1.0e30])
def test_kdot3(cond):
    x, y, ref, _ = accupy.generate_ill_conditioned_dot_product(100, cond)
    assert abs(accupy.kdot(x, y, K=3) - ref) < 1.0e-15 * abs(ref)
    return


@pytest.mark.parametrize("cond", [1.0, 1.0e15, 1.0e30, 1.0e38])
def test_fdot(cond):
    x, y, ref, _ = accupy.generate_ill_conditioned_dot_product(100, cond)
    assert abs(accupy.fdot(x, y) - ref) < 1.0e-15 * abs(ref)
    return


def test_accuracy_comparison_illcond(target_cond=None):
    if target_cond is None:
        target_cond = [10 ** k for k in range(2)]

    kernels = [
        numpy.dot,
        lambda x, y: accupy.kdot(x, y, K=2),
        lambda x, y: accupy.kdot(x, y, K=3),
        accupy.fdot,
    ]
    labels = ["numpy.dot", "accupy.kdot[2]", "accupy.kdot[3]", "accupy.fdot"]
    data = numpy.empty((len(target_cond), len(kernels)))
    condition_numbers = numpy.empty(len(target_cond))
    numpy.random.seed(0)
    for k, tc in enumerate(target_cond):
        x, y, ref, C = accupy.generate_ill_conditioned_dot_product(1000, tc)
        condition_numbers[k] = C
        data[k] = [abs(kernel(x, y) - ref) / abs(ref) for kernel in kernels]

    # sort
    s = numpy.argsort(condition_numbers)
    condition_numbers = condition_numbers[s]
    data = data[s]

    for label, d in zip(labels, data.T):
        plt.loglog(condition_numbers, d, label=label)

    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.grid()
    plt.ylim(5.0e-18, 1.0)
    plt.xlabel("condition number")
    plt.ylabel("relative error")
    plt.gca().set_aspect(1.3)

    # plt.show()
    # <https://stackoverflow.com/a/10154763/353337>
    plt.savefig(
        "accuracy-dot.svg",
        transparent=True,
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )
    return


def test_speed_comparison1(n_range=None):
    if n_range is None:
        n_range = [2 ** k for k in range(2)]

    numpy.random.seed(0)
    perfplot.plot(
        setup=lambda n: (numpy.random.rand(n, 100), numpy.random.rand(100, n)),
        kernels=[
            lambda xy: numpy.dot(*xy),
            lambda xy: accupy.kdot(*xy, K=2),
            lambda xy: accupy.kdot(*xy, K=3),
            lambda xy: accupy.fdot(*xy),
        ],
        labels=["numpy.dot", "accupy.kdot[2]", "accupy.kdot[3]", "accupy.fdot"],
        colors=plt.rcParams["axes.prop_cycle"].by_key()["color"][:4],
        n_range=n_range,
        title="dot(random(n, 100), random(100, n))",
        xlabel="n",
        logx=True,
        logy=True,
    )
    plt.gca().set_aspect(0.2)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    # plt.show()
    plt.savefig(
        "speed-comparison-dot1.svg",
        transparent=True,
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )
    return


def test_speed_comparison2(n_range=None):
    if n_range is None:
        n_range = [2 ** k for k in range(2)]

    numpy.random.seed(0)
    perfplot.plot(
        setup=lambda n: (numpy.random.rand(100, n), numpy.random.rand(n, 100)),
        kernels=[
            lambda xy: numpy.dot(*xy),
            lambda xy: accupy.kdot(*xy, K=2),
            lambda xy: accupy.kdot(*xy, K=3),
            lambda xy: accupy.fdot(*xy),
        ],
        labels=["numpy.dot", "accupy.kdot[2]", "accupy.kdot[3]", "accupy.fdot"],
        colors=plt.rcParams["axes.prop_cycle"].by_key()["color"][:4],
        n_range=n_range,
        title="dot(random(100, n), random(n, 100))",
        xlabel="n",
        logx=True,
        logy=True,
    )
    plt.gca().set_aspect(0.2)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    # plt.show()
    plt.savefig(
        "speed-comparison-dot2.svg",
        transparent=True,
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )
    return


def test_discontiguous():
    x = numpy.random.rand(3, 10)
    y = numpy.random.rand(3, 10)

    accupy.kdot(x.T, y)
    accupy.fdot(x.T, y)
    return


if __name__ == "__main__":
    # test_accuracy_comparison_illcond([10 ** k for k in range(0, 37, 1)])
    # test_speed_comparison1(n_range=[2**k for k in range(8)])
    test_speed_comparison2(n_range=[2 ** k for k in range(8)])

<p align="center">
  <a href="https://github.com/nschloe/accupy"><img alt="accupy" src="https://nschloe.github.io/accupy/logo-with-text.svg" width="40%"></a>
  <p align="center">Accurate sums and (dot) products for Python.</p>
</p>

[![PyPi Version](https://img.shields.io/pypi/v/accupy.svg?style=flat-square)](https://pypi.org/project/accupy)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/accupy.svg?style=flat-square)](https://pypi.org/pypi/accupy/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1185173.svg?style=flat-square)](https://doi.org/10.5281/zenodo.1185173)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/accupy.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/accupy)
[![PyPi downloads](https://img.shields.io/pypi/dm/accupy.svg?style=flat-square)](https://pypistats.org/packages/accupy)

[![gh-actions](https://img.shields.io/github/workflow/status/nschloe/accupy/ci?style=flat-square)](https://github.com/nschloe/accupy/actions?query=workflow%3Aci)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/accupy.svg?style=flat-square)](https://codecov.io/gh/nschloe/accupy)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

### Sums

Summing up values in a list can get tricky if the values are floating point
numbers; digit cancellation can occur and the result may come out wrong. A
classical example is the sum
```
1.0e16 + 1.0 - 1.0e16
```
The actual result is `1.0`, but in double precision, this will result in `0.0`.
While in this example the failure is quite obvious, it can get a lot more
tricky than that. accupy provides
```python
p, exact, cond = accupy.generate_ill_conditioned_sum(100, 1.0e20)
```
which, given a length and a target condition number, will produce an array of
floating point numbers that is hard to sum up.

accupy has the following methods for summation:

  * `accupy.kahan_sum(p)`: [Kahan
    summation](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)

  * `accupy.fsum(p)`: A vectorization wrapper around
    [math.fsum](https://docs.python.org/3/library/math.html#math.fsum) (which
    uses Shewchuck's algorithm [[1]](#references) (see also
    [here](https://code.activestate.com/recipes/393090/))).

  * `accupy.ksum(p, K=2)`: Summation in K-fold precision (from [[2]](#references))

All summation methods sum the first dimension of a multidimensional NumPy array.

Let's compare them.

#### Accuracy comparison (sum)

![](https://nschloe.github.io/accupy/accuracy-sums.svg)

As expected, the naive
[sum](https://docs.python.org/3/library/functions.html#sum) performs very badly
with ill-conditioned sums; likewise for
[`numpy.sum`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html)
which uses pairwise summation. Kahan summation not significantly better; [this,
too, is
expected](https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Accuracy).

Computing the sum with 2-fold accuracy in `accupy.ksum` gives the correct
result if the condition is at most in the range of machine precision; further
increasing `K` helps with worse conditions.

Shewchuck's algorithm in `math.fsum` always gives the correct result to full
floating point precision.


#### Runtime comparison (sum)

![](https://nschloe.github.io/accupy/speed-comparison1.svg)

![](https://nschloe.github.io/accupy/speed-comparison2.svg)

We compare more and more sums of fixed size (above) and larger and larger sums,
but a fixed number of them (below). In both cases, the least accurate method is
the fastest (`numpy.sum`), and the most accurate the slowest (`accupy.fsum`).

### Dot products

accupy has the following methods for dot products:

  * `accupy.fdot(p)`: A transformation of the dot product of length _n_ into a
    sum of length _2n_, computed with
    [math.fsum](https://docs.python.org/3/library/math.html#math.fsum)

  * `accupy.kdot(p, K=2)`: Dot product in K-fold precision (from
    [[2]](#references))

Let's compare them.

#### Accuracy comparison (dot)

accupy can construct ill-conditioned dot products with
```python
x, y, exact, cond = accupy.generate_ill_conditioned_dot_product(100, 1.0e20)
```
With this, the accuracy of the different methods is compared.

![](https://nschloe.github.io/accupy/accuracy-dot.svg)

As for sums, `numpy.dot` is the least accurate, followed by instanced of `kdot`.
`fdot` is provably accurate up into the last digit

#### Runtime comparison (dot)

![](https://nschloe.github.io/accupy/speed-comparison-dot1.svg)
![](https://nschloe.github.io/accupy/speed-comparison-dot2.svg)

NumPy's `numpy.dot` is _much_ faster than all alternatives provided by accupy.
This is because the bookkeeping of truncation errors takes more steps, but
mostly because of NumPy's highly optimized dot implementation.


### References

1. [Richard Shewchuk, _Adaptive Precision Floating-Point Arithmetic and Fast
   Robust Geometric Predicates_, J. Discrete Comput. Geom. (1997), 18(305),
   305–363](https://doi.org/10.1007/PL00009321)

2. [Takeshi Ogita, Siegfried M. Rump, and Shin'ichi Oishi, _Accurate Sum and Dot
   Product_, SIAM J. Sci. Comput. (2006), 26(6), 1955–1988 (34
   pages)](https://doi.org/10.1137/030601818)

### Dependencies

accupy needs the C++ [Eigen
library](http://eigen.tuxfamily.org/index.php?title=Main_Page), provided in
Debian/Ubuntu by
[`libeigen3-dev`](https://packages.ubuntu.com/search?keywords=libeigen3-dev).

### Installation

accupy is [available from the Python Package Index](https://pypi.org/project/accupy/), so with
```
pip install accupy
```
you can install.

### Testing

To run the tests, just check out this repository and type
```
MPLBACKEND=Agg pytest
```

### License
accupy is published under the [GPLv3+ license](https://www.gnu.org/licenses/gpl-3.0.en.html).

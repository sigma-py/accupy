# accupy

Accurate sums and products for Python.

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/accupy/master.svg)](https://circleci.com/gh/nschloe/accupy/tree/master)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/accupy.svg)](https://codecov.io/gh/nschloe/accupy)
[![awesome](https://img.shields.io/badge/awesome-yes-brightgreen.svg)](https://img.shields.io/badge/awesome-yes-brightgreen.svg)
[![PyPi Version](https://img.shields.io/pypi/v/accupy.svg)](https://pypi.python.org/pypi/accupy)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/accupy.svg?style=social&label=Stars)](https://github.com/nschloe/accupy)


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
p, exact = accupy.generate_ill_conditioned_sum(100, 1.0e20)
```
which given a length and a target condition number will produce an array if
floating point numbers that's hard to sum up.

accupy has the following methods for summation:

  * `accupy.kahan_sum(p)`: [Kahan
    summation](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)

  * `accupy.fsum(p)`: A thin wrapper around
    [math.fsum](https://docs.python.org/3/library/math.html#math.fsum) (which
    uses the [Shewchuck
    algorithm](https://code.activestate.com/recipes/393090/))

  * `accupy.ksum(p, K=2)`: Summation in K-fold precision (from [[1]](#references))

Let's compare the methods.

#### Accuracy comparison

<img src="https://nschloe.github.io/accupy/accuracy-comparison.png" width="40%">

For the 

### References

1. [Takeshi Ogita, Siegfried M. Rump, and Shin'ichi Oishi, _Accurate Sum and Dot
   Product_, SIAM J. Sci. Comput., 26(6), 1955–1988. (34
   pages).](https://doi.org/10.1137/030601818)

### Installation

accupy is [available from the Python Package Index](https://pypi.python.org/pypi/accupy/), so with
```
pip install -U accupy
```
you can install/upgrade.

### Testing

To run the tests, just check out this repository and type
```
MPLBACKEND=Agg pytest
```

### Distribution

To create a new release

1. bump the `__version__` number,

2. publish to PyPi and GitHub:
    ```
    $ make publish
    ```

### License
accupy is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).

# accupy

Accurate sums and products for Python.

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/accupy/master.svg)](https://circleci.com/gh/nschloe/accupy/tree/master)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/accupy.svg)](https://codecov.io/gh/nschloe/accupy)
[![awesome](https://img.shields.io/badge/awesome-yes-brightgreen.svg)](https://img.shields.io/badge/awesome-yes-brightgreen.svg)
[![PyPi Version](https://img.shields.io/pypi/v/accupy.svg)](https://pypi.python.org/pypi/accupy)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/accupy.svg?style=social&label=Stars)](https://github.com/nschloe/accupy)

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

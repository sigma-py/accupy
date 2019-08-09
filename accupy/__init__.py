from __future__ import print_function

from .__about__ import (
    __author__,
    __email__,
    __copyright__,
    __license__,
    __version__,
    __status__,
)

from .dot import kdot, fdot
from .ill_cond import generate_ill_conditioned_sum, generate_ill_conditioned_dot_product
from .sums import knuth_sum, decker_sum, distill, ksum, fsum, kahan_sum

__all__ = [
    "__author__",
    "__email__",
    "__copyright__",
    "__license__",
    "__version__",
    "__status__",
    "kdot",
    "fdot",
    "generate_ill_conditioned_sum",
    "generate_ill_conditioned_dot_product",
    "knuth_sum",
    "decker_sum",
    "distill",
    "ksum",
    "fsum",
    "kahan_sum",
]

try:
    import pipdate
except ImportError:
    pass
else:
    if pipdate.needs_checking(__name__):
        print(pipdate.check(__name__, __version__), end="")

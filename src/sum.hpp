#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Algorithm 4.3. Error-free vector transformation for summation.
//
// The vector p is transformed without changing the sum, and p_n is replaced
// by float(sum(p)). Kahan [21] calls this a "distillation algorithm".
//
// Keep an eye on <https://github.com/pybind/pybind11/issues/1294> for a
// potentially better treatment of the dimensionality.
void
distill(py::array_t<double, py::array::c_style | py::array::forcecast> p) {
  auto r = p.mutable_unchecked<2>();
  for (ssize_t i = 1; i < r.shape(0); i++) {
    for (ssize_t j = 0; j < r.shape(1); j++) {
      double x = r(i, j) + r(i-1, j);
      double z = x - r(i, j);
      double y = (r(i, j) - (x-z)) + (r(i-1, j)-z);
      r(i, j) = x;
      r(i-1, j) = y;
    }
  }
}

double
kahan(py::array_t<double, py::array::c_style | py::array::forcecast> p) {
  // Kahan summation.
  // See <https://en.wikipedia.org/wiki/Kahan_summation_algorithm> for
  // details.
  auto r = p.unchecked<1>();

  double s = 0.0;
  double c = 0.0;
  for (ssize_t i = 0; i < r.shape(0); i++) {
    double y = r(i) - c;
    double t = s + y;
    c = (t-s) - y;
    s = t;
  }
  return s;
}

double
neumaier(py::array_t<double, py::array::c_style | py::array::forcecast> p) {
  // Neumaier summation
  // <https://en.wikipedia.org/wiki/Kahan_summation_algorithm#Further_enhancements>.
  auto r = p.unchecked<1>();

  double s = r(0);
  double c = 0.0;
  for (ssize_t i = 1; i < r.shape(0); i++) {
    double t = s + r(i);
    if (std::fabs(s) > std::fabs(r(i))) {
      c += (s - t) + r(i);
    } else {
      c += (r(i) - t) + s;
    }
    s = t;
  }
  return s + c;
}


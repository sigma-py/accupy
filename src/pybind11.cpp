#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Algorithm 4.3. Error-free vector transformation for summation.
//
// The vector p is transformed without changing the sum, and p_n is replaced
// by float(sum(p)). Kahan [21] calls this a "distillation algorithm".
//
// For now, we'll have to keep two versions around; see
// <https://github.com/pybind/pybind11/issues/1294>.
py::array_t<double>
distill1(py::array_t<double, py::array::c_style | py::array::forcecast> p) {
  auto r = p.mutable_unchecked<1>();
  for (ssize_t i = 1; i < r.shape(0); i++) {
    double x = r(i) + r(i-1);
    double z = x - r(i);
    double y = (r(i) - (x-z)) + (r(i-1)-z);
    r(i) = x;
    r(i-1) = y;
  }
  return p;
}

py::array_t<double>
distill2(py::array_t<double, py::array::c_style | py::array::forcecast> p) {
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
  return p;
}

PYBIND11_MODULE(_accupy, m) {
  m.def("distill1", &distill1);
  m.def("distill2", &distill2);
}

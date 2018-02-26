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
      double y = (r(i, j) - (x-z)) + (r(i-1, j) - z);
      r(i, j) = x;
      r(i-1, j) = y;
    }
  }
}

py::array_t<double>
kahan(py::array_t<double, py::array::c_style | py::array::forcecast> p) {
  // Kahan summation.
  // See <https://en.wikipedia.org/wiki/Kahan_summation_algorithm> for
  // details.
  auto buf_p = p.request();
  if (buf_p.ndim != 2)
    throw std::runtime_error("Number of dimensions must be two");

  const ssize_t m = buf_p.shape[0];
  const ssize_t n = buf_p.shape[1];

  auto s = py::array_t<double>(n);
  auto buf_s = s.request();

  auto c = py::array_t<double>(n);
  auto buf_c = c.request();

  double *ptr_p = (double *) buf_p.ptr;
  double *ptr_c = (double *) buf_c.ptr;
  double *ptr_s = (double *) buf_s.ptr;

  // zero out c and s
  std::fill(ptr_c, ptr_c+n, 0.0);
  std::fill(ptr_s, ptr_s+n, 0.0);

  // Kahan
  for (ssize_t i = 0; i < m; i++) {
    for (ssize_t j = 0; j < n; j++) {
      double y = ptr_p[i*n + j] - ptr_c[j];
      double t = ptr_s[j] + y;
      ptr_c[j] = (t - ptr_s[j]) - y;
      ptr_s[j] = t;
    }
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
    c += (std::fabs(s) > std::fabs(r(i))) ? (s - t) + r(i) : (r(i) - t) + s;
    s = t;
  }
  return s + c;
}


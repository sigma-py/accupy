#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <unsupported/Eigen/CXX11/Tensor>

namespace py = pybind11;
using namespace pybind11::literals;


// =============================================================================
// sum.h
// =============================================================================
// void
// distill(py::array_t<double, py::array::c_style | py::array::forcecast> p) {
//   auto r = p.mutable_unchecked<2>();
//   for (ssize_t i = 1; i < r.shape(0); i++) {
//     for (ssize_t j = 0; j < r.shape(1); j++) {
//       double x = r(i, j) + r(i-1, j);
//       double z = x - r(i, j);
//       double y = (r(i, j) - (x-z)) + (r(i-1, j) - z);
//       r(i, j) = x;
//       r(i-1, j) = y;
//     }
//   }
// }

using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Algorithm 4.3. Error-free vector transformation for summation.
//
// The vector p is transformed without changing the sum, and p_n is replaced
// by float(sum(p)). Kahan [21] calls this a "distillation algorithm".
void
distill(Eigen::Ref<RowMatrixXd> r) {
  for (int i = 1; i < r.rows(); i++) {
    auto x = r.row(i) + r.row(i-1);
    auto z = x - r.row(i);
    for (int j = 0; j < r.cols(); j++) {
      const double xj = x(j);
      r(i-1, j) = (r(i, j) - (x(j) - z(j))) + (r(i-1, j) - z(j));
      r(i, j) = xj;
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
// =============================================================================
// dot.h
// Headers aren't automatically installed though;
// <https://github.com/pypa/packaging-problems/issues/84>.
// =============================================================================
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

py::array_t<double>
kdot_helper(Eigen::Ref<RowMatrixXd> x, Eigen::Ref<RowMatrixXd> y) {
  // Algorithm 5.10. Dot product algorithm in K-fold working precision, K >= 3.
  if (x.cols() != y.rows())
    throw std::runtime_error("Input shapes must match");

  const int n = x.cols();

  // Use Eigen::Tensor to avoid stack overflows with native C arrays.
  Eigen::Tensor<double, 4, Eigen::RowMajor> result(2, n, x.rows(), y.cols());

  // After the loop, p will hold the naive values of the dot product, result(0)
  // the multiplication errors and result(1) the addition errors.
  auto p = RowMatrixXd(x.rows(), y.cols());
  p.setZero();

  // Use ikj ordering for speed; see, e.g.,
  // <https://software.intel.com/en-us/articles/putting-your-data-and-code-in-order-optimization-and-memory-part-1>
  for (int i=0; i < x.rows(); i++) {
    for (int k=0; k < n; k++) {
      for (int j=0; j < y.cols(); j++) {
        // product with exact error
        double h = x(i, k) * y(k, j);
        result(0, k, i, j) = fma(x(i, k), y(k, j), -h);
        // Knuth sum: p+h with exact error z2
        double z0 = p(i, j) + h;
        double z1 = z0 - p(i, j);
        double z2 = (p(i, j) - (z0-z1)) + (h-z1);
        p(i, j) = z0;
        result(1, k, i, j) = z2;
      }
    }
  }

  // Override the meaningless first addition error; it's exactly 0.0 anyways.
  for (int i=0; i < x.rows(); i++)
    for (int j=0; j < y.cols(); j++)
      result(1, 0, i, j) = p(i, j);

  return py::array_t<double>(
      std::vector<ptrdiff_t>{2, n, x.rows(), y.cols()},
      result.data()
      );
}
// =============================================================================
PYBIND11_MODULE(_accupy, m) {
  // sum:
  m.def("distill", &distill, "r"_a.noconvert());
  m.def("kahan", &kahan, "p"_a.noconvert());
  // dot:
  m.def("kdot_helper", &kdot_helper, "x"_a.noconvert(), "y"_a.noconvert());
}

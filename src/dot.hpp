#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <unsupported/Eigen/CXX11/Tensor>

namespace py = pybind11;
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

py::array_t<double>
kdot_helper(Eigen::Ref<RowMatrixXd> x, Eigen::Ref<RowMatrixXd> y) {
  // Algorithm 5.10. Dot product algorithm in K-fold working precision, K >= 3.
  if (x.cols() != y.rows())
    throw std::runtime_error("Input shapes must match");

  const int n = x.cols();

  // Use Eigen::Tensor to avoid stack overflows with native C arrays.
  Eigen::Tensor<double, 4, Eigen::RowMajor> result(2, n, x.rows(), y.cols());

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
        // knuth sum p+h with exact error z2
        double z0 = p(i, j) + h;
        double z1 = z0 - p(i, j);
        double z2 = (p(i, j) - (z0-z1)) + (h-z1);
        p(i, j) = z0;
        result(1, k, i, j) = z2;
      }
    }
  }

  for (int i=0; i < x.rows(); i++)
    for (int j=0; j < y.cols(); j++)
      result(1, 0, i, j) = p(i, j);

  return py::array_t<double>(
      std::vector<ptrdiff_t>{2, n, x.rows(), y.cols()},
      result.data()
      );
}

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

  for (int i=0; i < x.rows(); i++) {
    for (int j=0; j < y.cols(); j++) {
      double p = x(i, 0) * y(0, j);
      result(0, 0, i, j) = fma(x(i, 0), y(0, j), -p);

      for (ssize_t k=1; k < n; k++) {
        // product with exact error
        double h = x(i, k) * y(k, j);
        result(0, k, i, j) = fma(x(i, k), y(k, j), -h);
        // knuth sum
        double z0 = p + h;
        double z1 = z0 - p;
        double z2 = (p - (z0-z1)) + (h-z1);
        p = z0;
        result(1, k-1, i, j) = z2;
      }
      result(1, n-1, i, j) = p;
    }
  }

  return py::array_t<double>(
      std::vector<ptrdiff_t>{2, n, x.rows(), y.cols()},
      result.data()
      );
}

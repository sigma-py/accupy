#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

RowMatrixXd
kdot_helper(
  py::array_t<double, py::array::c_style | py::array::forcecast> x,
  py::array_t<double, py::array::c_style | py::array::forcecast> y
  ) {
  // Algorithm 5.10. Dot product algorithm in K-fold working precision, K >= 3.
  auto buf1 = x.request();
  auto buf2 = y.request();

  if (buf1.ndim != 1 || buf2.ndim != 1)
    throw std::runtime_error("Number of dimensions must be 1");

  if (buf1.size != buf2.size)
    throw std::runtime_error("Input shapes must match");

  // No pointer is passed, so NumPy will allocate the buffer
  // auto result = py::array_t<double>(2 * buf1.size);
  auto result = RowMatrixXd(2, buf1.size);

  double *ptr1 = (double *) buf1.ptr;
  double *ptr2 = (double *) buf2.ptr;

  double p = ptr1[0] * ptr2[0];
  result(0, 0) = fma(ptr1[0], ptr2[0], -p);

  for (ssize_t idx=1; idx < buf1.size; idx++) {
    // product with exact error
    double h = ptr1[idx] * ptr2[idx];
    result(0, idx) = fma(ptr1[idx], ptr2[idx], -h);
    // knuth sum
    double z0 = p + h;
    double z1 = z0 - p;
    double z2 = (p - (z0-z1)) + (h-z1);
    p = z0;
    result(1, idx-1) = z2;
  }
  result(1, buf1.size-1) = p;

  return result;
}

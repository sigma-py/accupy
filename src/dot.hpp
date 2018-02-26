#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double>
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

  /* No pointer is passed, so NumPy will allocate the buffer */
  auto result = py::array_t<double>(2 * buf1.size);

  auto buf3 = result.request();

  double *ptr1 = (double *) buf1.ptr;
  double *ptr2 = (double *) buf2.ptr;
  double *ptr3 = (double *) buf3.ptr;

  double p = ptr1[0] * ptr2[0];
  ptr3[0] = fma(ptr1[0], ptr2[0], -p);

  for (ssize_t idx=1; idx < buf1.size; idx++) {
    // product with exact error
    double h = ptr1[idx] * ptr2[idx];
    ptr3[idx] = fma(ptr1[idx], ptr2[idx], -h);
    // knuth sum
    double z0 = p + h;
    double z1 = z0 - p;
    double z2 = (p - (z0-z1)) + (h-z1);
    p = z0;
    ptr3[buf1.size+idx-1] = z2;
  }
  ptr3[2*buf1.shape[0] - 1] = p;

  return result;
}

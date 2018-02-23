#include "sum.hpp"
#include "dot.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


PYBIND11_MODULE(_accupy, m) {
  m.def("distill", &distill);
  m.def("kahan", &kahan);
  m.def("neumaier", &neumaier);
  m.def("kdot_helper", &kdot_helper);
}

#include "sum.hpp"
#include "dot.hpp"

#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(_accupy, m) {
  m.def("distill", &distill);
  m.def("kahan", &kahan);
  m.def("neumaier", &neumaier);
  m.def("kdot_helper", &kdot_helper);
  m.def("distill", &distill, "r"_a.noconvert());
}

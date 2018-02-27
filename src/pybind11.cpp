#include "sum.hpp"
#include "dot.hpp"

#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(_accupy, m) {
  // sum:
  m.def("distill", &distill, "r"_a.noconvert());
  m.def("kahan", &kahan, "p"_a.noconvert());
  // dot:
  m.def("kdot_helper", &kdot_helper, "x"_a.noconvert(), "y"_a.noconvert());
}

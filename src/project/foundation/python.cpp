#include "../../../include/foundation/example.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// NOTE: This sets compile time level. In addition, you need to set the
// runtime level low enough to show these (e.g. trace for everything)
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO

#include <fstream>
#include <spdlog/spdlog.h>

namespace py = pybind11;

using namespace foundation;

PYBIND11_MODULE(foundation, m) {
  m.doc() = R"pbdoc(
        Bindings to the foundation.
        ---------------------------
    )pbdoc";

  m.def(
      "set_spdlog_level", [](const std::string &level) { spdlog::set_level(spdlog::level::from_str(level)); },
      "Set spd log level. Supported levels are: trace, debug, info, warn, error, critical, off.");

  m.def("add", add, "Add two numbers", py::arg("a"), py::arg("b"), R"pbdoc(
        Add two numbers
  )pbdoc");
}

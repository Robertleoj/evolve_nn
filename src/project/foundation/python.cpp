#include "foundation/graph/compiled_graph.hpp"
#include "foundation/graph/nodes.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

#include <fstream>
#include <iostream>
#include <string>

namespace py = pybind11;

using namespace foundation;
using namespace graph;

void add_tensors(torch::Tensor a) {
  torch::Tensor b = torch::randn({2, 3});
  torch::Tensor c = torch::randn({2, 3});
  std::cout << a + b << std::endl;
}

// pybind module
PYBIND11_MODULE(foundation, m) {
  m.doc() = "python bindings"; // optional module docstring
  m.def("add_tensors", &add_tensors, "Add two tensors");

  // graph submodule
  py::module graphMod = m.def_submodule("graph", "Graph module");

  // Binding node classes with constructors

  py::class_<Node, std::shared_ptr<Node>>(graphMod, "Node");

  py::class_<DataNode, Node, std::shared_ptr<DataNode>>(graphMod, "DataNode");

  py::class_<InputNode, DataNode, std::shared_ptr<InputNode>>(graphMod, "InputNode").def(py::init<>());

  py::class_<ResponseInputNode, DataNode, std::shared_ptr<ResponseInputNode>>(graphMod, "ResponseInputNode")
      .def(py::init<>());

  py::class_<OutputNode, DataNode, std::shared_ptr<OutputNode>>(graphMod, "OutputNode").def(py::init<>());

  py::class_<LossOutputNode, DataNode, std::shared_ptr<LossOutputNode>>(graphMod, "LossOutputNode").def(py::init<>());

  py::class_<ParameterNode, DataNode, std::shared_ptr<ParameterNode>>(graphMod, "ParameterNode").def(py::init<>());

  py::class_<OperatorNode, Node, std::shared_ptr<OperatorNode>>(graphMod, "OperatorNode");

  py::class_<AddNode, OperatorNode, std::shared_ptr<AddNode>>(graphMod, "AddNode").def(py::init<>());

  py::class_<NegNode, OperatorNode, std::shared_ptr<NegNode>>(graphMod, "NegNode").def(py::init<>());

  py::class_<ProdNode, OperatorNode, std::shared_ptr<ProdNode>>(graphMod, "ProdNode").def(py::init<>());

  py::class_<GELUNode, OperatorNode, std::shared_ptr<GELUNode>>(graphMod, "GELUNode").def(py::init<>());

  py::class_<LogNode, OperatorNode, std::shared_ptr<LogNode>>(graphMod, "LogNode").def(py::init<>());

  py::class_<ExpNode, OperatorNode, std::shared_ptr<ExpNode>>(graphMod, "ExpNode").def(py::init<>());

  py::class_<CompiledGraphWrapper>(graphMod, "CompiledGraph")
      .def(py::init<std::vector<std::shared_ptr<Node>>, std::vector<std::vector<int>>, std::vector<int>,
                    std::vector<int>>(),
           py::arg("nodes_topsorted"), py::arg("rev_adj_list"), py::arg("input_order"), py::arg("output_order"))
      .def("forward", &CompiledGraphWrapper::forward);
}

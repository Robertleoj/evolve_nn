import numpy as np
import project.graph.compiled as compiled_
import project.graph.graph as graph_

init_spec = {
    "node_specs": [
        {"name": "input"},
        {"name": "add"},
        {"name": "output"},
    ],
    "rev_adj_list": [[], [0, 0], [1]],
    "input_node_order": [0],
    "output_node_order": [2],
}


x = np.linspace(0, 1, 100)
y_clean = np.sin(x * np.pi * 2)
y = y_clean + 0.05 * np.random.randn(*x.shape)


g = graph_.make_graph(**init_spec)


comp_cpp = compiled_.to_cpp_compiled(g)

print("Forwarding")
comp_cpp.forward([x])

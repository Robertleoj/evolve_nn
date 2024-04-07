# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: project-xOhHZUaJ-py3.10
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
import torch

import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm
import matplotlib.pyplot as plt
from project.graph_novec import graph, nodes, compiled, mutation

# %%
graph_spec = {
    "node_names": [
        "input",
        "parameter",
        "prod",
        "parameter",
        "add",
        "output",
    ],
    "edge_list": [
        (0, 2), 
        (1, 2), 
        (2, 4),
        (3, 4), 
        (4, 5)
    ],
}

# %%
a = torch.tensor([1, 2])
b = torch.tensor([2, 3])
torch.stack([a, b], dim=-1)

# %%
net = graph.make_graph(**graph_spec)
graph.show_graph(net)

# %%
compiled_net = compiled.CompiledGraph.from_graph(net)
compiled.show_compiled(compiled_net)
compiled_net([torch.tensor([1, 2, 3])])

# %%
a = 3
b = 2

x = torch.linspace(0, 1, 100)
y_clean = a * x + b + 0.1
y = y_clean + 0.1 * torch.randn(x.size())

plt.scatter(x, y)
plt.plot(x, y_clean, color="red")



# %%
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(compiled_net.parameters(), lr=1e-3, momentum=0.9)
x_in = rearrange(x, "b -> b")
targ = rearrange(y, "b -> b")

num_epochs = 1000
pbar = tqdm(total=num_epochs)
for i in range(num_epochs):
    optimizer.zero_grad()
    output = compiled_net([x_in])[0]
    loss = loss_fn(output, targ)
    loss.backward()
    optimizer.step()

    # Update tqdm every iteration with loss info in the bar
    pbar.set_description(f"Loss: {loss.item():.4f}")
    pbar.update(1)
pbar.close()

# %%
with torch.no_grad():
    y_hat = compiled_net([x_in])[0]
    print(y_hat)


print(y_hat.shape)
y_hat

# %%
plt.plot(x, y_hat, color="green")
plt.scatter(x, y)



# %%
mutated = net
graph.show_graph(mutated)

# %%

mutated, changed = mutation.expand_edge(mutated)
print(f"Expanded edge: {changed}")
mutated, changed = mutation.add_edge(mutated)
print(f"Added edge: {changed}")

graph.show_graph(mutated)

# %%
mut_compiled = compiled.CompiledGraph.from_graph(mutated)

# %%
compiled.show_compiled(mut_compiled)

# %%
mut_compiled([torch.tensor([1, 2, 3])])

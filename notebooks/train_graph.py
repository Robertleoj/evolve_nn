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

# %% [markdown]
# # Test training of computational graphs

# %% [markdown]
# ## 1D Linear regression

# %%
# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt
import torch
from einops import rearrange
from project.graph.graph import CompiledGraph, make_graph, show_compiled, show_graph
from tqdm import tqdm

# %%
a = 3
b = 2

x = torch.linspace(0, 1, 100)
y_clean = a * x + b + 0.1
y = y_clean + 0.1 * torch.randn(x.size())

plt.scatter(x, y)
plt.plot(x, y_clean, color="red")

# %%
x_dim = 1
graph_spec = {
    "node_specs": [
        {"name": "input"},
        {"name": "parameter"},
        {"name": "prod"},
        {"name": "parameter"},
        {"name": "add"},
        {"name": "output"},
    ],
    "rev_adj_list": [[], [], [0, 1], [], [2, 3], [4]],
    "input_node_order": [0],
    "output_node_order": [5],
}

# %%
net_graph = make_graph(**graph_spec)
show_graph(net_graph)

# %%
compiled = CompiledGraph.from_graph(net_graph)
show_compiled(compiled)

# %%
compiled.stored_parameters

# %%
with torch.no_grad():
    display(compiled(torch.tensor([[1.0]])))

# %%
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(compiled.parameters(), lr=1e-2)

# %%

x_in = rearrange(x, "b -> b 1 1")
targ = rearrange(y, "b -> b 1 1")

num_epochs = 1000
pbar = tqdm(total=num_epochs)
for i in range(num_epochs):
    optimizer.zero_grad()
    output = compiled([x_in])[0]
    loss = loss_fn(output, targ)
    loss.backward()
    optimizer.step()

    # Update tqdm every iteration with loss info in the bar
    pbar.set_description(f"Loss: {loss.item():.4f}")
    pbar.update(1)
pbar.close()

# %%
y_hat = rearrange(compiled([x_in])[0], "b 1 1 -> b").detach().numpy()
y_hat

# %%
plt.plot(x, y_hat, color="green")
plt.scatter(x, y)

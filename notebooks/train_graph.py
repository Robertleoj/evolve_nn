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
from project.graph import graph
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
        {"name": "input", "shape": (x_dim, 1)},
        {
            "name": "parameter",
            "shape": (
                x_dim,
                x_dim,
            ),
        },
        {"name": "matmul", "input_shapes": [(x_dim, x_dim), (x_dim, 1)], "shape": (x_dim, 1)},
        {"name": "parameter", "shape": (x_dim, 1)},
        {"name": "add", "input_shapes": [(x_dim, 1), (x_dim, 1)], "shape": (x_dim, 1)},
        {"name": "output", "shape": (x_dim, 1)},
    ],
    "edge_list": [(0, 2), (1, 2), (2, 4), (3, 4), (4, 5)],
    "index_map": {
        (0, 2): 1,
        (1, 2): 0,
        (2, 4): 1,
        (3, 4): 0,
    },
}

# %%
net_graph = graph.Graph(**graph_spec)
net_graph.show()

# %%
compiled = graph.CompiledGraph.from_graph(net_graph)

# %%
compiled._parameters

# %%
compiled(torch.tensor([[1.0]]))

# %%
loss_fn = torch.nn.MSELoss()

# %%
optimizer = torch.optim.SGD(compiled.parameters(), lr=1e-3, momentum=0.9)

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

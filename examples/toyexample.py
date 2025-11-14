#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import torch
from smalldiffusion import (
    DatasaurusDozen,
    ScheduleLogLinear,
    Swissroll,
    TimeInputMLP,
    samples,
    training_loop,
)
from torch.utils.data import DataLoader

import wandb

wandb.init(project="hf-diffusion-study")


def plot_batch(batch):
    fig, ax = plt.subplots()
    batch = batch.cpu().numpy()
    ax.scatter(batch[:, 0], batch[:, 1], marker=".")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Batch")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


# # Data
# Load and plot 2D point data from the [Datasaurus Dozen](https://jumpingrivers.github.io/datasauRus/) dataset.

# In[2]:


from pathlib import Path

dataset_path = Path(__file__).parent.parent / "datasets" / "DatasaurusDozen.tsv"

# Try replacing dataset with 'dino', 'bullseye', 'h_lines', 'x_shape', etc.
dataset = DatasaurusDozen(csv_file=dataset_path, dataset="dino")
# Or use the SwissRoll dataset
# dataset = Swissroll(np.pi/2, 5*np.pi, 100)
loader = DataLoader(dataset, batch_size=2130)
fig = plot_batch(next(iter(loader)))
wandb.log({"batch": fig})


# # Schedule
# Use a log-linear $\sigma$ schedule with 200 steps.

# In[14]:


schedule = ScheduleLogLinear(N=200, sigma_min=0.01, sigma_max=10)
fig, ax = plt.subplots()
ax.plot(schedule.sigmas)
ax.set_xlabel("$t$")
ax.set_ylabel("$\sigma_t$")
ax.set_yscale("log")
wandb.log({"schedule": fig})


# The $\sigma$ schedule is embedded sinusoidally as $[\sin(\log(\sigma)/2), \cos(\log(\sigma)/2)]$.

# In[15]:


from smalldiffusion.model import get_sigma_embeds

sx, sy = get_sigma_embeds(len(schedule), schedule.sigmas).T
fig, ax = plt.subplots()
plt.plot(sx, label="$\sin(\log(\sigma_t)/2)$")
ax.plot(sy, label="$\cos(\log(\sigma_t)/2)$")
ax.set_xlabel("$t$")
ax.legend()
wandb.log({"sigma_embeds": fig})


# # Model
# Define a simple diffusion model using a MLP. The 4-dimensional input to this MLP
# is the $\sigma$ embeddings concatenated with $x$.
# The MLP has a 2-dimensional output, the predicted noise $\epsilon$.

# In[5]:


model = TimeInputMLP(hidden_dims=(16, 128, 128, 128, 128, 16))
print(model)


# # Train
# Train the diffusion model and plot training loss

# In[ ]:


trainer = training_loop(loader, model, schedule, epochs=15000, lr=1e-3)
losses = []
for ns in trainer:
    loss_item = ns.loss.item()
    losses.append(loss_item)
    wandb.log({"loss": loss_item})


# In[10]:


fig, ax = plt.subplots()
ax.plot(moving_average(losses, 100))
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training Loss")
ax.grid(True, alpha=0.3)
plt.tight_layout()
wandb.log({"training_loss": fig})


# # Sample
# Sample from the diffusion model using 20 sampling steps, with gradient estimation sampler

# In[8]:


# For DDPM sampling, change to gam=1, mu=0.5
# For DDIM sampling, change to gam=1, mu=0
*xts, x0 = samples(model, schedule.sample_sigmas(20), batchsize=1500, gam=2, mu=0)
fig = plot_batch(x0)
wandb.log({"sample": fig})

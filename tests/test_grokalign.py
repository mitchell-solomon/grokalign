import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import types
import numpy as np
import jax.numpy as jnp
from jax import random
import pytest

import jax_utils
import utils

import torch
import torch.nn as nn


def _constant_proj_torch(self, batch_size, output_dim):
    v = torch.ones(self.num_projections, batch_size, output_dim, device=self.device)
    return v / torch.sqrt(torch.tensor(float(output_dim)))


def _constant_proj_jax(self, batch_size, output_dim):
    v = jnp.ones((self.num_projections, batch_size, output_dim))
    return v / jnp.sqrt(float(output_dim))


@pytest.mark.parametrize("batch", [2])
def test_grokalign_linear_equivalence(monkeypatch, batch):
    W = np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=np.float32)
    torch_model = nn.Linear(2, 3, bias=False)
    torch_model.weight.data = torch.tensor(W)
    ga_torch = utils.GrokAlign(torch_model, num_projections=1)

    def apply_fn(params, x):
        return jnp.dot(x, params["w"].T)

    params = {"w": jnp.array(W)}
    ga_jax = jax_utils.GrokAlign(apply_fn, num_projections=1, key=random.PRNGKey(0))

    monkeypatch.setattr(utils.GrokAlign, "_get_random_projections", _constant_proj_torch)
    monkeypatch.setattr(jax_utils.GrokAlign, "_get_random_projections", _constant_proj_jax)

    x_np = np.array([[0.5, 1.0], [1.5, -0.5]], dtype=np.float32)[:batch]
    x_torch = torch.tensor(x_np)
    x_jax = jnp.array(x_np)

    torch_val = ga_torch(x_torch).item()
    jax_val = float(ga_jax(params, x_jax))

    assert np.allclose(torch_val, jax_val, atol=1e-6)


def test_gradfilter_ema_equivalence():
    model = nn.Linear(2, 2, bias=False)
    model.weight.grad = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

    grads_jax = {"weight": jnp.array([[0.1, 0.2], [0.3, 0.4]])}

    ema_torch = utils.gradfilter_ema(model, grads=None, alpha=0.98, lamb=2.0)
    grads_jax_out, ema_jax = jax_utils.gradfilter_ema(grads_jax, ema=None, alpha=0.98, lamb=2.0)

    assert np.allclose(ema_torch["weight"].numpy(), np.array(ema_jax["weight"]))
    assert np.allclose(model.weight.grad.numpy(), np.array(grads_jax_out["weight"]))

    model.weight.grad = torch.tensor([[0.2, 0.1], [0.4, 0.3]])
    grads_jax = {"weight": jnp.array([[0.2, 0.1], [0.4, 0.3]])}

    ema_torch = utils.gradfilter_ema(model, grads=ema_torch, alpha=0.98, lamb=2.0)
    grads_jax_out, ema_jax = jax_utils.gradfilter_ema(grads_jax, ema=ema_jax, alpha=0.98, lamb=2.0)

    assert np.allclose(ema_torch["weight"].numpy(), np.array(ema_jax["weight"]))
    assert np.allclose(model.weight.grad.numpy(), np.array(grads_jax_out["weight"]))

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import types
import numpy as np
import jax.numpy as jnp
import pytest

import jax_utils
import utils

import torch
import torch.nn as nn


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

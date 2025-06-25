import numpy as np
import jax
import jax.numpy as jnp
from jax import random, tree_util, vmap
from jax.example_libraries import stax
import torch


def set_seed(seed: int):
    """Seed numpy and JAX PRNG and return a PRNG key."""
    np.random.seed(seed)
    return random.PRNGKey(seed)


def get_mnist_loaders(train_points, test_points, batch_size, data_dir="./data"):
    """Return simple MNIST loaders yielding JAX arrays."""
    import torchvision
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        torchvision.transforms.Lambda(lambda x: x.view(-1)),
    ])
    train_set = torchvision.datasets.MNIST(root=data_dir, train=True,
                                           transform=transform, download=True)
    test_set = torchvision.datasets.MNIST(root=data_dir, train=False,
                                          transform=transform, download=True)

    train_subset = torch.utils.data.Subset(train_set, range(train_points))
    test_subset = torch.utils.data.Subset(test_set, range(test_points))

    def collate(batch):
        xs, ys = zip(*batch)
        x = jnp.stack([jnp.array(i.numpy()) for i in xs])
        y = jnp.array([int(i) for i in ys])
        return x, y

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate)

    return {"train": train_loader, "test": test_loader}


def build_model(config, key):
    layers = [stax.Dense(config.width), stax.Relu]
    for _ in range(config.depth - 2):
        layers += [stax.Dense(config.width), stax.Relu]
    layers += [stax.Dense(10)]

    init_fn, apply_fn = stax.serial(*layers)
    out_shape, params = init_fn(key, (config.batch_size, 784))
    if getattr(config, "init_scale", 1.0) != 1.0:
        params = tree_util.tree_map(lambda p: p * config.init_scale, params)
    return params, apply_fn


def compute_accuracy(params, apply_fn, loader):
    acc = []
    for x, labels in loader:
        preds = jnp.argmax(apply_fn(params, x), axis=-1)
        acc.append(jnp.mean(preds == labels))
    return float(jnp.mean(jnp.array(acc)))


class GrokAlign:
    def __init__(self, apply_fn, num_projections=1, key=None):
        self.apply_fn = apply_fn
        self.num_projections = num_projections
        self.key = random.PRNGKey(0) if key is None else key

    def _get_random_projections(self, batch_size, output_dim):
        self.key, subkey = random.split(self.key)
        v = random.normal(subkey, (self.num_projections, batch_size, output_dim))
        v = v / jnp.clip(jnp.linalg.norm(v, axis=-1, keepdims=True), 1e-8)
        return v

    def compute_jacobian_norm(self, params, x):
        def model_fn(inputs):
            return self.apply_fn(params, inputs)

        output = model_fn(x)
        batch_size, output_dim = output.shape
        v = self._get_random_projections(batch_size, output_dim)

        def jvp_single(v_proj):
            _, Jv = jax.jvp(model_fn, (x,), (v_proj,))
            Jv_flat = Jv.reshape(Jv.shape[0], -1)
            return jnp.sum(Jv_flat ** 2, axis=1)

        norm_sum = jnp.sum(vmap(jvp_single)(v), axis=0)
        scaling = output_dim / self.num_projections
        return jnp.sqrt(jnp.clip(norm_sum * scaling, 1e-8))

    def __call__(self, params, inputs):
        return jnp.mean(self.compute_jacobian_norm(params, inputs))


def gradfilter_ema(grads, ema=None, alpha=0.98, lamb=2.0):
    if ema is None:
        ema = grads
    else:
        ema = tree_util.tree_map(lambda e, g: e * alpha + g * (1 - alpha), ema, grads)
    grads = tree_util.tree_map(lambda g, e: g + e * lamb, grads, ema)
    return grads, ema

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, tree_util, vmap
from jax.example_libraries import stax
from functools import partial



def set_seed(seed: int):
    """Seed numpy and JAX PRNG and return a PRNG key."""
    np.random.seed(seed)
    return random.PRNGKey(seed)


def get_mnist_loaders(train_points, test_points, batch_size, data_dir="./data"):
    """Return MNIST loaders with JITed preprocessing using TFDS."""
    import tensorflow_datasets as tfds

    mean = jnp.array(0.1307, dtype=jnp.float32)
    std = jnp.array(0.3081, dtype=jnp.float32)

    def _preprocess_img(img):
        img = img.astype(jnp.float32) / 255.0
        img = (img - mean) / std
        return jnp.reshape(img, -1)

    preprocess_batch = jax.jit(vmap(_preprocess_img))

    def load_split(split, points):
        ds = tfds.load(
            "mnist",
            split=f"{split}[:{points}]",
            batch_size=-1,
            data_dir=data_dir,
        )
        images = preprocess_batch(jnp.array(ds["image"].numpy()))
        labels = jnp.array(ds["label"].numpy(), dtype=jnp.int32)
        return images, labels

    class SimpleLoader:
        def __init__(self, x, y, batch):
            self.x = x
            self.y = y
            self.batch = batch

        @partial(jax.jit, static_argnums=0)
        def _get_batch(self, start):
            end = start + self.batch
            return self.x[start:end], self.y[start:end]

        def __iter__(self):
            for i in range(0, self.x.shape[0], self.batch):
                yield self._get_batch(i)

    x_train, y_train = load_split("train", train_points)
    x_test, y_test = load_split("test", test_points)

    train_loader = SimpleLoader(x_train, y_train, batch_size)
    test_loader = SimpleLoader(x_test, y_test, batch_size)

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
    """Compute accuracy over a loader using a single batched call."""
    xs, ys = [], []
    for x, labels in loader:
        xs.append(x)
        ys.append(labels)

    if len(xs) == 0:
        return 0.0

    x = jnp.concatenate(xs, axis=0)
    y = jnp.concatenate(ys, axis=0)

    @jax.jit
    def _predict(p, inputs):
        return apply_fn(p, inputs)

    preds = jnp.argmax(_predict(params, x), axis=-1)
    return float(jnp.mean(preds == y))


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

    @jax.jit
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

    @jax.jit
    def __call__(self, params, inputs):
        return jnp.mean(self.compute_jacobian_norm(params, inputs))


@jax.jit
def gradfilter_ema(grads, ema=None, alpha=0.98, lamb=2.0):
    if ema is None:
        ema = grads
    else:
        ema = tree_util.tree_map(
            lambda e, g: e * alpha + g * (1 - alpha), ema, grads
        )
    grads = tree_util.tree_map(lambda g, e: g + e * lamb, grads, ema)
    return grads, ema

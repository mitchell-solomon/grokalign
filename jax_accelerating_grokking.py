import argparse
import time

import jax
import jax.numpy as jnp
import optax
import wandb as wb
from tqdm import tqdm

from functools import partial

from jax_utils import (
    set_seed,
    get_mnist_loaders,
    build_model,
    compute_accuracy,
    # GrokAlign,
    grokalign_loss,
    gradfilter_ema,
)


def train(config, data_dir="./data"):
    key = set_seed(config.seed)

    loaders = get_mnist_loaders(
        config.train_points, config.test_points, config.batch_size, data_dir
    )
    params, apply_fn = build_model(config, key)

    optimizer = optax.adamw(config.lr, weight_decay=config.weight_decay)
    opt_state = optimizer.init(params)


    @partial(jax.jit, static_argnums=(7, 8, 9))
    def train_step(
        params, opt_state, ema, key, x, labels, sigma,
        loss_fn_type, lambda_jac, use_grokfast
    ):

        if config.adv_training:
            key, subkey = jax.random.split(key)
            x = x + jax.random.normal(subkey, x.shape) * sigma

        def loss_fn(p, inp, lab, jac_key):
            logits = apply_fn(p, inp)
            if loss_fn_type == "CrossEntropy":
                loss = optax.softmax_cross_entropy_with_integer_labels(logits, lab).mean()
            else:
                one_hot = jax.nn.one_hot(lab, 10)
                loss = jnp.mean((logits - one_hot) ** 2)

            if lambda_jac > 0.0:
                loss += grokalign_loss(apply_fn, p, inp, jac_key, lambda_jac, num_projections=1)

            return loss


        key, jac_key = jax.random.split(key)
        loss, grads = jax.value_and_grad(loss_fn)(params, x, labels, jac_key)

        if use_grokfast:
            grads, ema = gradfilter_ema(grads, ema=ema, alpha=0.8, lamb=0.1)


        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, ema, key, loss



    # grokalign = GrokAlign(apply_fn) if config.lambda_jac > 0.0 else None
    ema = None

    run_name = (
        f"{config.loss_fn}-"
        f"{'GF' if config.grokfast else ''}"
        f"{'GA' if config.lambda_jac > 0 else ''}"
        f"{'Wd' if config.weight_decay > 0 else ''}"
        f"{'At' if config.adv_training else ''}-"
        f"{config.seed}"
    )
    run = wb.init(project="accelerating_grokking", config=vars(config), name=run_name)

    total_time = 0.0
    pbar = tqdm(range(config.steps + 1))
    for step in pbar:
        start = time.time()
        if config.adv_training:
            train_acc = compute_accuracy(params, apply_fn, loaders["train"])
            sigma = max(0.06 * (1 - train_acc), 0.03)
        else:
            sigma = 0.0

        for x, labels in loaders["train"]:
            params, opt_state, ema, key, loss = train_step(
                params, opt_state, ema, key, x, labels, sigma,
                config.loss_fn, config.lambda_jac, config.grokfast
            )


        total_time += time.time() - start
        pbar.set_description(f"{float(loss):.4f}")

        if step % 10 == 0:
            stats = {
                "step": step,
                "loss": float(loss),
                "optimization_time": total_time,
            }
            for split in ["train", "test"]:
                stats[f"{split}_accuracy"] = compute_accuracy(
                    params, apply_fn, loaders[split]
                )
            wb.log(stats)
            if stats["test_accuracy"] > 0.85:
                break

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_points", type=int, default=1024)
    parser.add_argument("--test_points", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lambda_jac", type=float, default=0.0)
    parser.add_argument("--grokfast", action="store_true")
    parser.add_argument("--adv_training", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--init_scale", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=196)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument(
        "--loss_fn", type=str, choices=["MSE", "CrossEntropy"], default="MSE"
    )
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    train(args)

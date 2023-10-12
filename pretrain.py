import json
import logging
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import bert
import helpers
import wandb

sec_per_hr = 60 * 60
log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("pretrain")

cfg = helpers.DotDict(
    # Data
    data_root="/local/scratch/openwebtext",
    mask_rate=0.2,
    # Train
    train_steps=200_000,
    eval_steps=32,
    eval_every=1000,
    lr=1e-4,
    batch_size=64,
    # General
    seed=0,
    log_every=10,
    save_every=50_000,
    ckpt_root="/local/scratch/stevens.994/mlm",
    max_hrs=24,
)

model_cfg = bert.Config(
    vocab_size=50258,
    max_length=512,
    n_embd=768,
    n_layers=12,
    n_heads=12,
    dropout=0.0,  # only one pass through data
)


key = jax.random.PRNGKey(cfg.seed)
rng = np.random.default_rng(cfg.seed)


class DataLoader:
    def __init__(self, cfg, seq_length, split, mask_rate, *, rng):
        self.batch_size = cfg.batch_size
        self.seq_length = seq_length

        self.split = split
        self.mask_rate = mask_rate

        self.enc = bert.get_enc()
        self.mask_token = self.enc._special_tokens["<|mask|>"]

        self.data = np.memmap(f"{cfg.data_root}/{split}.bin", dtype=np.uint16, mode="r")

        self.rng = rng
        self.max_start = len(self.data) - self.seq_length

    def mask(self, toks):
        toks = toks.copy()
        idx = self.rng.uniform(size=toks.shape) < self.mask_rate
        toks[idx] = self.mask_token
        return toks, idx

    def __iter__(self):
        while True:
            start_toks = self.rng.integers(
                0, self.max_start, (self.batch_size,), dtype=np.int64
            )
            y = np.stack([self.data[i : i + self.seq_length] for i in start_toks])
            x, idx = self.mask(y)

            yield x, y, idx


def val(model, dataloader):
    @eqx.filter_jit
    def compute_loss(model, x, y, masks):
        logits = jax.vmap(model)(x)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        return jnp.sum(losses * masks) / masks.sum()

    losses = jnp.zeros((cfg.eval_steps,))
    for step, (x, y, masks) in zip(range(cfg.eval_steps), dataloader):
        loss = compute_loss(model, x, y, masks)
        losses = losses.at[step].set(loss)

    return jnp.mean(losses)


def main():
    # Set up model
    model_key, _ = jax.random.split(key)
    model = bert.Transformer(model_cfg, key=model_key)

    # Count params
    def count(module):
        params = eqx.filter(module, eqx.is_array)
        return sum(x.size for x in jax.tree_util.tree_leaves(params))

    n_params = count(model) - count(model.wte) - count(model.wpe)
    logger.info(json.dumps(dict(n_params=helpers.human(n_params))))
    wandb.config.n_params = n_params

    optim = optax.adamw(cfg.lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    train_dataloader = DataLoader(
        cfg, model_cfg.max_length, "train", cfg.mask_rate, rng=rng
    )
    val_dataloader = DataLoader(
        cfg, model_cfg.max_length, "val", cfg.mask_rate, rng=rng
    )

    @eqx.filter_value_and_grad
    def compute_loss(model, x, y, masks):
        logits = jax.vmap(model)(x)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        return jnp.sum(losses * masks) / masks.sum()

    @eqx.filter_jit
    def make_step(model, x, y, masks, opt_state):
        loss, grads = compute_loss(model, x, y, masks)
        updates, opt_state = optim.update(grads, opt_state, params=model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    for step, (x, y, masks) in zip(range(start, cfg.train_steps), train_dataloader):
        loss, model, opt_state = make_step(model, x, y, masks, opt_state)

        metrics = {}

        if step % cfg.log_every == 0:
            metrics.update(
                {
                    "perf/step": step,
                    "perf/toks": (step + 1) * cfg.batch_size * model_cfg.max_length,
                    "train/loss": loss.item(),
                    "train/lr": cfg.lr,
                }
            )

            # Check if we should stop training (hours)
            elapsed = time.time() - start_time
            if elapsed > cfg.max_hrs * sec_per_hr:
                logger.info("Stopping training after %.1f hours.", elapsed / sec_per_hr)
                break

        if step % cfg.eval_every == 0:
            val_loss = val(model, val_dataloader)
            metrics.update({"val/loss": val_loss.item()})

        if step % cfg.save_every == 0:
            helpers.save(f"{cfg.ckpt_root}/{run.id}/step{step}", model_cfg, model)

        if metrics:
            logger.info(json.dumps(metrics))
            wandb.log(metrics)
            metrics = {}

    helpers.save(f"{cfg.ckpt_root}/{run.id}/step{step}", model_cfg, model)
    if metrics:
        logger.info(json.dumps(metrics))
        wandb.log(metrics)


if __name__ == "__main__":
    start = 0
    start_time = time.time()

    run = wandb.init(
        project="mlm-cramming",
        entity="samuelstevens",
        config=cfg,
        job_type="pretrain",
        resume=False,
        tags=[],
    )

    main()

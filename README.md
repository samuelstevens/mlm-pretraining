# smol-lang

Simple code to train encoder-only masked language models (like BERT) on GPUs.

I am reimplementing [Geiping et al. 2022](https://arxiv.org/abs/2212.14034) (*Cramming: Training a Language Model on a Single GPU in One Day*) using [JAX](https://jax.readthedocs.io/en/latest/index.html) and [Equinox](https://docs.kidger.site/equinox/).

The goal is to better understand BERT-pretraining from scratch.
1 step of the baseline takes around 1 second.
I'm training for 100K steps by default, 100K / 60 s/min / 60 min/hr = 28 hours.
With that in mind, I hope to start a new job every day on an A6000, introducing changes one by one.
## Install

I used CUDA 12.2 on my lab's Nvidia A6000 GPUs:

```sh
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Run

```sh
XLA_PYTHON_CLIENT_MEM_FRACTION=.99 CUDA_VISIBLE_DEVICES=1 python pretrain.py
```

## Data

I am using openwebtext, downloaded using Huggingface's datasets libary.

## To Do

* [ ] Switch to gated linear unit
* [ ] Couple the input/output embeddings
* [ ] Resuming from checkpoint. See the [Equinox docs](https://docs.kidger.site/equinox/examples/serialisation/) for how to load models.
* [ ] Evaluating on downstream tasks
* [ ] Log gradient norm
* [x] Learning rate scheduler
* [x] Weight decay
* [x] Gradient clipping
* [x] Adam betas and epsilon
* [x] Logging
* [x] Tag different versions with git.
* [x] Add a final layernorm to stabilize training
* [x] Disable linear layer biases
* [x] Disable QKV biases
* [x] Checkpointing
* [x] Stop after 24 hours
* [x] Track FLOP utilization
* [x] Mixed precision (see [jmp](https://github.com/google-deepmind/jmp) and [this Equinox issue](https://github.com/patrick-kidger/equinox/issues/221))
  * Might need loss scaling
* [x] Gradient accumulation

## Out of Scope

* Multi-GPU parallelism

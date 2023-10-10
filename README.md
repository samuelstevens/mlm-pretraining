# smol-lang

Simple code to train encoder-only masked language models (like BERT) on GPUs.

I am reimplementing [Geiping et al. 2022](https://arxiv.org/abs/2212.14034) (*Cramming: Training a Language Model on a Single GPU in One Day) using [jax](https://jax.readthedocs.io/en/latest/index.html).

## Install

I used CUDA 12.2 on my lab's Nvidia A6000 GPUs:

```sh
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Data

I am using openwebtext, downloaded using Huggingface's datasets libary.

## To Do

* [ ] Disable linear layer biases
* [ ] Disable QKV biases
* [ ] Switch to gated linear unit
* [ ] Couple the input/output embeddings
* [ ] Add a final layernorm to stabilize training
* [ ] Gradient accumulation
* [ ] Learning rate scheduler
* [ ] Weight decay
* [ ] Gradient clipping
* [ ] Adam betas and epsilon
* [x] Logging
* [ ] Checkpointing
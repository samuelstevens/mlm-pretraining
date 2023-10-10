import dataclasses
import functools as ft

import equinox as eqx
import jax
import jax.numpy as jnp
import tiktoken.load


@dataclasses.dataclass(frozen=True)
class Config:
    vocab_size: int
    max_length: int
    n_embd: int
    n_layers: int
    n_heads: int
    dropout: float


class Block(eqx.Module):
    mhsa: eqx.nn.MultiheadAttention
    ln1: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP
    ln2: eqx.nn.LayerNorm

    def __init__(self, cfg, *, key: jax.random.PRNGKey):
        mhsa_key, mlp_key = jax.random.split(key, num=2)

        self.ln1 = eqx.nn.LayerNorm(cfg.n_embd)
        self.mhsa = eqx.nn.MultiheadAttention(
            cfg.n_heads, cfg.n_embd, dropout_p=cfg.dropout, key=mhsa_key
        )

        self.ln2 = eqx.nn.LayerNorm(cfg.n_embd)

        hidden_dim = int(4 * cfg.n_embd)

        self.mlp = eqx.nn.MLP(
            in_size=cfg.n_embd,
            out_size=cfg.n_embd,
            width_size=hidden_dim,
            depth=1,
            activation=ft.partial(jax.nn.gelu, approximate=True),
            key=mlp_key,
        )

    def __call__(self, x, *, dropout=False, key=None):
        attn_key, dropout_key = (None, None) if key is None else jax.random.split(key)

        x_ln = jax.vmap(self.ln1)(x)
        x = x + self.mhsa(query=x_ln, key_=x_ln, value=x_ln, key=attn_key)

        x_ln = jax.vmap(self.ln2)(x)
        x = x + jax.vmap(self.mlp)(x_ln, key=dropout_key)

        return x


class Transformer(eqx.Module):
    wte: eqx.nn.Embedding
    wpe: eqx.nn.Embedding
    layers: eqx.nn.Sequential
    head: eqx.nn.Linear

    def __init__(self, cfg, *, key: jax.random.PRNGKey):
        wte_key, wpe_key, head_key, *layer_key = jax.random.split(
            key, num=cfg.n_layers + 3
        )
        self.wte = eqx.nn.Embedding(cfg.vocab_size, cfg.n_embd, key=wte_key)
        self.wpe = eqx.nn.Embedding(cfg.max_length, cfg.n_embd, key=wpe_key)
        self.layers = eqx.nn.Sequential([Block(cfg, key=key) for key in layer_key])
        self.head = eqx.nn.Linear(cfg.n_embd, cfg.vocab_size, key=head_key)

    def __call__(self, toks, *, dropout=False, key=None):
        pos = jnp.arange(0, toks.shape[0])

        tok_emb = jax.vmap(self.wte)(toks)
        pos_emb = jax.vmap(self.wpe)(pos)
        emb = tok_emb + pos_emb

        x = self.layers(emb)

        x = jax.vmap(self.head)(x)

        return x


def get_enc():
    mergeable_ranks = tiktoken.load.data_gym_to_mergeable_bpe_ranks(
        vocab_bpe_file="https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
        encoder_json_file="https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
    )

    return tiktoken.core.Encoding(
        name="gpt2",
        explicit_n_vocab=50258,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=mergeable_ranks,
        special_tokens={"<|endoftext|>": 50256, "<|mask|>": 50257},
    )

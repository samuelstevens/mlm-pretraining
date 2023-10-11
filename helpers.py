import dataclasses
import json
import os

import equinox as eqx


def save(filename, model_cfg, model):
    cfg_dct = dataclasses.asdict(model_cfg)
    cfg_dct["model"] = "bert"

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(f"{filename}-hparams.json", "w") as fd:
        json.dump(cfg_dct, fd)

    with open(f"{filename}-model.bin", "wb") as fd:
        eqx.tree_serialise_leaves(fd, model)


class DotDict(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise ValueError(attr)


def human(num):
    prefix = "-" if num < 0 else ""

    num = abs(num)

    for i, suffix in [(1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if num > i:
            return f"{prefix}{num / i:.1f}{suffix}"

    if num < 1:
        return f"{prefix}{num:.3g}"

    return f"{prefix}{num}"

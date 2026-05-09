"""NNsight ``Submodule`` wrapper for residual streams.

Wraps a transformer block so we can read/write its activation tensor through
NNsight's tracer with a uniform interface, regardless of whether the layer
returns a tensor or a tuple.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch as t


@dataclass(frozen=True)
class Submodule:
    name: str
    submodule: Any
    use_input: bool = False
    is_tuple: bool = False

    def __hash__(self) -> int:
        return hash(self.name)

    def get_activation(self):
        out = self.submodule.input if self.use_input else self.submodule.output
        return out[0] if self.is_tuple else out

    def set_activation(self, x) -> None:
        if self.use_input:
            target = self.submodule.input
        else:
            target = self.submodule.output
        if self.is_tuple:
            target[0][:] = x
        else:
            target[:] = x

    def stop_grad(self) -> None:
        if self.use_input:
            target = self.submodule.input
        else:
            target = self.submodule.output
        if self.is_tuple:
            target[0].grad = t.zeros_like(target[0])
        else:
            target.grad = t.zeros_like(target)

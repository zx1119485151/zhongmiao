# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Star block used by the experimental ``model/*_SCS.yaml`` configs."""

from __future__ import annotations

import torch
from torch import nn

from .conv import Conv, DWConv


class StarBlock(nn.Module):
    """StarNet-style block: depthwise filter, multiplicative branch, pointwise projection.

    ``c2`` defaults to ``c1`` so both YAML spellings work: ``[32, 32, 1]`` and ``[32]``.
    """

    def __init__(self, c1: int, c2: int | None = None, expansion: float = 1.0) -> None:
        super().__init__()
        c2 = c1 if c2 is None else c2
        hidden = max(8, int(round(c2 * expansion)))
        self.dw = DWConv(c1, c1, 3, 1)
        self.branch_a = Conv(c1, hidden, 1, 1)
        self.branch_b = Conv(c1, hidden, 1, 1)
        self.project = Conv(hidden, c2, 1, 1, act=False)
        self.act = nn.SiLU()
        self.add = c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.branch_a(y) * self.branch_b(y)
        y = self.project(y)
        return self.act(y + x) if self.add else self.act(y)

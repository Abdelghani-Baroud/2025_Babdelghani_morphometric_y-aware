# models/tabular_resnet_mlp.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResMLPBlock(nn.Module):
    """
    PreNorm Residual MLP block (tabular ResNet style):
      x -> LN -> Linear -> GELU -> Dropout -> Linear -> Dropout -> + x
    """
    def __init__(self, dim: int, hidden_mult: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * hidden_mult)

        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h


class TabularResNetMLPEncoder(nn.Module):
    """
    Tabular ResNet-style MLP encoder for continuous vectors (e.g., PCA latents).
    Outputs an L2-normalized embedding for CLIP-style alignment.

    Args:
      in_dim:  dimension of tabular input (PCA dim)
      out_dim: embedding dimension (must match vision encoder output dim)
      width:   hidden width for the residual trunk
      depth:   number of residual blocks
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 512,
        depth: int = 4,
        hidden_mult: float = 2.0,
        dropout: float = 0.0,
        in_norm: bool = True,
        out_norm: bool = True,
    ):
        super().__init__()

        self.in_norm = nn.LayerNorm(in_dim) if in_norm else None

        self.fc_in = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList(
            [ResMLPBlock(width, hidden_mult=hidden_mult, dropout=dropout) for _ in range(depth)]
        )

        self.trunk_norm = nn.LayerNorm(width) if out_norm else nn.Identity()
        self.fc_out = nn.Linear(width, out_dim)

        # Optional: small init helps stability
        nn.init.trunc_normal_(self.fc_in.weight, std=0.02)
        nn.init.zeros_(self.fc_in.bias)
        nn.init.trunc_normal_(self.fc_out.weight, std=0.02)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_dim]
        returns: [B, out_dim] L2-normalized
        """
        if self.in_norm is not None:
            x = self.in_norm(x)

        h = self.fc_in(x)  # [B, width]
        for blk in self.blocks:
            h = blk(h)

        h = self.trunk_norm(h)
        z = self.fc_out(h)  # [B, out_dim]
        z = F.normalize(z, dim=-1)
        return z

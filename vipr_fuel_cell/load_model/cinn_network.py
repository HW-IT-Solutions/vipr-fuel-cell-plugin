"""Inference-only cINN architecture compatible with the published checkpoints."""

from __future__ import annotations

import torch
from torch import nn


class PEMFCCINN(nn.Module):
    """Glow-style conditional INN used for PEMFC parameter reconstruction."""

    def __init__(
        self,
        parameter_names: list[str],
        condition_names: list[str],
        n_blocks: int = 10,
        subnet_hidden_size: int = 512,
    ):
        super().__init__()
        self.parameter_names = list(parameter_names)
        self.condition_names = list(condition_names)
        self.input_dim = len(parameter_names)
        self.condition_dim = len(condition_names)
        self.n_blocks = int(n_blocks)
        self.subnet_hidden_size = int(subnet_hidden_size)
        self.cinn = self._build_inn()

    def _build_inn(self):
        try:
            import FrEIA.framework as Ff
            import FrEIA.modules as Fm
        except ImportError as exc:
            raise ImportError(
                "PEMFC cINN loading requires FrEIA. Install vipr-fuel-cell with its dependencies."
            ) from exc

        hidden_size = self.subnet_hidden_size

        def subnet(ch_in, ch_out):
            return nn.Sequential(
                nn.Linear(ch_in, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, ch_out),
            )

        nodes = [Ff.InputNode(self.input_dim, name="input")]
        condition = Ff.ConditionNode(self.condition_dim, name="condition")
        for block_index in range(self.n_blocks):
            nodes.append(
                Ff.Node(
                    nodes[-1],
                    Fm.GLOWCouplingBlock,
                    {"subnet_constructor": subnet, "clamp": 1.2},
                    conditions=condition,
                    name=f"coupling_block_{block_index}",
                )
            )
            nodes.append(
                Ff.Node(
                    nodes[-1],
                    Fm.PermuteRandom,
                    {"seed": block_index},
                    name=f"permute_{block_index}",
                )
            )
        nodes.append(Ff.OutputNode(nodes[-1], name="output"))
        return Ff.ReversibleGraphNet(nodes + [condition], verbose=False)

    def inverse(self, latent: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Map latent samples and scaled sensor conditions to scaled parameters."""
        parameters, _ = self.cinn(latent.float(), c=conditions.float(), rev=True)
        return parameters

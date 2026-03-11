from __future__ import annotations

from dataclasses import dataclass, field

from energnn.graph import GraphStructure
from energnn.model.ready_to_use import TinyRecurrentEquivariantGNN


@dataclass
class LoadFlowModelConfig:
    n_breakpoints: int = 50
    latent_dimension: int = 16
    hidden_sizes: list[int] = field(default_factory=lambda: [32])
    n_steps: int = 20
    seed: int = 0


def build_loadflow_model(
    *,
    in_structure: GraphStructure,
    out_structure: GraphStructure,
    cfg: LoadFlowModelConfig,
) -> TinyRecurrentEquivariantGNN:
    return TinyRecurrentEquivariantGNN(
        in_structure=in_structure,
        out_structure=out_structure,
        seed=cfg.seed,
    )

from __future__ import annotations

from dataclasses import dataclass, field

from flax import nnx

from energnn.graph import GraphStructure
from energnn.model import (
    LocalSumMessageFunction,
    MLP,
    MLPEncoder,
    MLPEquivariantDecoder,
    RecurrentCoupler,
    SimpleGNN,
    TDigestNormalizer,
)


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
) -> SimpleGNN:
    """Build a modular SimpleGNN stack for load-flow supervision."""
    rngs = nnx.Rngs(cfg.seed)

    normalizer = TDigestNormalizer(in_structure=in_structure, n_breakpoints=cfg.n_breakpoints, update_limit=1000)

    encoder = MLPEncoder(
        in_structure=in_structure,
        hidden_sizes=cfg.hidden_sizes,
        activation=nnx.leaky_relu,
        out_size=cfg.latent_dimension,
        use_bias=True,
        final_activation=None,
        rngs=rngs,
    )

    message_function = LocalSumMessageFunction(
        in_graph_structure=in_structure,
        in_array_size=cfg.latent_dimension,
        hidden_sizes=cfg.hidden_sizes,
        activation=nnx.leaky_relu,
        out_size=cfg.latent_dimension,
        use_bias=True,
        final_activation=None,
        outer_activation=nnx.tanh,
        encoded_feature_size=cfg.latent_dimension,
        rngs=rngs,
    )

    phi = MLP(
        in_size=cfg.latent_dimension,
        hidden_sizes=[],
        activation=nnx.leaky_relu,
        out_size=cfg.latent_dimension,
        use_bias=True,
        final_activation=nnx.tanh,
        rngs=rngs,
    )

    coupler = RecurrentCoupler(
        phi=phi,
        message_functions=[message_function],
        n_steps=cfg.n_steps,
    )

    decoder = MLPEquivariantDecoder(
        in_graph_structure=in_structure,
        in_array_size=cfg.latent_dimension,
        hidden_sizes=cfg.hidden_sizes,
        activation=nnx.leaky_relu,
        out_structure=out_structure,
        use_bias=True,
        final_activation=None,
        encoded_feature_size=cfg.latent_dimension,
        rngs=rngs,
    )

    return SimpleGNN(
        normalizer=normalizer,
        encoder=encoder,
        coupler=coupler,
        decoder=decoder,
    )

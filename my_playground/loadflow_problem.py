from copy import deepcopy
from typing import Any
from collections.abc import Iterator
import math
import pickle
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from energnn.graph.edge import Edge
from energnn.graph.graph import Graph
from energnn.graph import EdgeStructure, GraphShape, GraphStructure, collate_graphs, max_shape
from energnn.graph.jax import JaxGraph, JaxEdge
from energnn.problem import Problem, ProblemBatch, ProblemLoader
from energnn.problem.metadata import ProblemMetadata
from utils.loadflow_data_utils import load_problem_from_pandapower_net


class LoadFlowProblem(Problem):
    CONTEXT_STRUCTURE: GraphStructure = GraphStructure(
        edges={
            "lines": EdgeStructure.from_list(
                address_list=["from_bus", "to_bus"],
                feature_list=["kind", "r_pu", "x_pu", "b_pu", "tap", "phase_shift_deg", "rating_pu"],
            ),
            "generators": EdgeStructure.from_list(
                address_list=["bus"],
                feature_list=[
                    "generator_type",  # PV vs slack
                    "p_set_pu",
                    "q_set_pu",
                    "vm_pu_set",
                    "va_deg_set",
                    "vm_mask",
                    "va_mask",
                    "p_mask",
                    "q_mask",
                    "p_min_pu",
                    "p_max_pu",
                    "q_min_pu",
                    "q_max_pu",
                ],
            ),
            "loads": EdgeStructure.from_list(
                address_list=["bus"],
                feature_list=["p_set_pu", "q_set_pu"],
            ),
        }
    )

    DECISION_STRUCTURE: GraphStructure = GraphStructure(
        edges={
            "lines": EdgeStructure.from_list(
                address_list=None,
                feature_list=["p_from_pu", "q_from_pu", "p_to_pu", "q_to_pu"],
            ),
            "generators": EdgeStructure.from_list(
                address_list=None,
                feature_list=["p_pu", "q_pu", "vm_pu", "va_deg"],
            ),
            "loads": EdgeStructure.from_list(
                address_list=None,
                feature_list=["p_pu", "q_pu", "vm_pu", "va_deg"],
            ),
        }
    )

    def __init__(
        self,
        context: Graph | None = None,
        oracle: Graph | None = None,
        *,
        net: Any | None = None,
    ):
        if context is None or oracle is None:
            if net is None:
                raise ValueError("Provide either (context, oracle) or net.")

            class _GraphCarrier:
                def __init__(self, *, context: Graph, oracle: Graph):
                    self.context = context
                    self.oracle = oracle

            carrier = load_problem_from_pandapower_net(net=net, problem_cls=_GraphCarrier)
            context = carrier.context
            oracle = carrier.oracle

        self._name = "load-flow-problem"
        self._context = context
        self._oracle = oracle
        self.jax_context = JaxGraph.from_numpy_graph(context)
        self.jax_oracle = JaxGraph.from_numpy_graph(oracle)
        self._decision_grad_fn = jax.grad(lambda pred: self.loss(prediction=pred))

    @classmethod
    def load_dataset(cls, dataset_path: str | Path) -> list[Any]:
        path = Path(dataset_path)
        with path.open("rb") as f:
            payload = pickle.load(f)

        if not isinstance(payload, dict):
            raise ValueError("Dataset file must contain a dictionary payload.")
        if payload.get("format") != "loadflow_3bus_dataset_pickle_v1":
            raise ValueError("Unsupported dataset format. Expected 'loadflow_3bus_dataset_pickle_v1'.")
        if "nets" not in payload or not isinstance(payload["nets"], list):
            raise ValueError("Dataset payload must contain a list field 'nets'.")

        return payload["nets"]

    def get_context(self, get_info: bool = False):
        info = {}
        if get_info:
            generator_features = self._context.edges["generators"].feature_dict
            info = {
                "n_addresses": int(self._context.true_shape.addresses),
                "edges": {k: int(v) for k, v in self._context.true_shape.edges.items()},
                "gen_vm_mask_ratio": float(np.mean(generator_features["vm_mask"])),
                "gen_va_mask_ratio": float(np.mean(generator_features["va_mask"])),
                "gen_p_mask_ratio": float(np.mean(generator_features["p_mask"])),
                "gen_q_mask_ratio": float(np.mean(generator_features["q_mask"])),
            }
        return self.jax_context, info

    @staticmethod
    def _jax_feature_idx_map(edge: JaxEdge) -> dict[str, int]:
        if edge.feature_names is None:
            return {}
        return {name: int(idx) for name, idx in edge.feature_names.items()}

    def _edge_gradient_and_mse(
        self,
        prediction_edge: JaxEdge,
        target_edge: JaxEdge,
    ) -> tuple[jax.Array, jax.Array]:
        idx_pred = self._jax_feature_idx_map(prediction_edge)
        idx_tgt = self._jax_feature_idx_map(target_edge)

        sq_error_sum = jnp.array(0.0)
        n_values = jnp.array(0.0)

        for feature_name, i_pred in idx_pred.items():
            if feature_name not in idx_tgt:
                continue
            i_tgt = idx_tgt[feature_name]
            pred = prediction_edge.feature_array[..., i_pred]
            target = target_edge.feature_array[..., i_tgt]
            diff = pred - target
            n_local = jnp.maximum(jnp.array(diff.size, dtype=diff.dtype), jnp.array(1.0, dtype=diff.dtype))
            sq_error_sum = sq_error_sum + jnp.sum(diff**2)
            n_values = n_values + n_local

        return sq_error_sum, n_values

    def loss(self, *, prediction: JaxGraph, target: JaxGraph | None = None) -> jax.Array:
        """Compute supervised MSE over all overlapping edge features.

        This is the scalar objective used both for gradient computation and metrics.
        """
        if target is None:
            target = self.jax_oracle

        sq_error_sum = jnp.array(0.0)
        n_values = jnp.array(0.0)

        for edge_name, prediction_edge in prediction.edges.items():
            if edge_name not in target.edges:
                raise ValueError(f"Missing edge '{edge_name}' in target graph.")
            target_edge = target.edges[edge_name]
            edge_sq_error_sum, edge_n_values = self._edge_gradient_and_mse(prediction_edge, target_edge)
            sq_error_sum = sq_error_sum + edge_sq_error_sum
            n_values = n_values + edge_n_values

        return sq_error_sum / jnp.maximum(n_values, 1.0)

    def get_gradient(self, *, decision, get_info: bool = False):
        gradient = self._decision_grad_fn(decision)

        info = {}
        if get_info:
            mse = self.loss(prediction=decision)
            info = {"mse": mse, "rmse": jnp.sqrt(mse)}
        return gradient, info

    def get_metrics(self, *, decision, get_info: bool = False):
        mse = self.loss(prediction=decision)
        rmse = jnp.sqrt(mse)

        info = {}
        if get_info:
            info = {"mse": mse, "rmse": rmse}

        return np.atleast_1d(np.asarray(rmse, dtype=np.float32)), info

    def get_metadata(self):
        context_shape = self._context.true_shape.to_jsonable_dict()
        decision_shape = {
            "edges": {k: int(v) for k, v in self._oracle.true_shape.edges.items()},
            "addresses": int(self._oracle.true_shape.addresses),
        }
        return ProblemMetadata(
            name=self._name,
            config_id="threebus-demo",
            code_version=2,
            context_shape=context_shape,
            decision_shape=decision_shape,
        )

    def save(self, *, path: str) -> None:
        raise NotImplementedError("Use pickle dataset files as source data. Save is handled by the data-generation script.")

    @property
    def context_structure(self) -> GraphStructure:
        return self.CONTEXT_STRUCTURE

    @property
    def decision_structure(self) -> GraphStructure:
        return self.DECISION_STRUCTURE


class LoadFlowBatch(ProblemBatch):
    def __init__(
        self,
        problems: list[LoadFlowProblem],
        context_max_shape: GraphShape | None = None,
        oracle_max_shape: GraphShape | None = None,
    ):
        if not problems:
            raise ValueError("LoadFlowBatch requires at least one problem instance.")

        self.problems = problems

        np_context_list = [pb.jax_context.to_numpy_graph() for pb in self.problems]
        np_oracle_list = [pb.jax_oracle.to_numpy_graph() for pb in self.problems]

        if context_max_shape is None:
            context_max_shape = max_shape([g.true_shape for g in np_context_list])
        if oracle_max_shape is None:
            oracle_max_shape = max_shape([g.true_shape for g in np_oracle_list])

        for g in np_context_list:
            g.pad(context_max_shape)
        for g in np_oracle_list:
            g.pad(oracle_max_shape)

        np_context_batch = collate_graphs(np_context_list)
        np_oracle_batch = collate_graphs(np_oracle_list)

        self.jax_context_batch = JaxGraph.from_numpy_graph(np_context_batch)
        self.jax_oracle_batch = JaxGraph.from_numpy_graph(np_oracle_batch)
        self._batch_grad_fn = jax.grad(lambda pred: self.loss(prediction=pred))

    @property
    def context_structure(self) -> GraphStructure:
        return LoadFlowProblem.CONTEXT_STRUCTURE

    @property
    def decision_structure(self) -> GraphStructure:
        return LoadFlowProblem.DECISION_STRUCTURE

    @staticmethod
    def _jax_feature_idx_map(edge: JaxEdge) -> dict[str, int]:
        if edge.feature_names is None:
            return {}
        return {name: int(idx) for name, idx in edge.feature_names.items()}

    def _edge_gradient_and_mse(self, prediction_edge: JaxEdge, target_edge: JaxEdge) -> tuple[jax.Array, jax.Array]:
        idx_pred = self._jax_feature_idx_map(prediction_edge)
        idx_tgt = self._jax_feature_idx_map(target_edge)

        sq_error_sum = jnp.array(0.0)
        n_values = jnp.array(0.0)

        for feature_name, i_pred in idx_pred.items():
            if feature_name not in idx_tgt:
                continue
            i_tgt = idx_tgt[feature_name]
            pred = prediction_edge.feature_array[..., i_pred]
            target = target_edge.feature_array[..., i_tgt]
            diff = pred - target
            n_local = jnp.maximum(jnp.array(diff.size, dtype=diff.dtype), jnp.array(1.0, dtype=diff.dtype))
            sq_error_sum = sq_error_sum + jnp.sum(diff**2)
            n_values = n_values + n_local

        return sq_error_sum, n_values

    def loss(self, *, prediction: JaxGraph, target: JaxGraph | None = None) -> jax.Array:
        if target is None:
            target = self.jax_oracle_batch

        sq_error_sum = jnp.array(0.0)
        n_values = jnp.array(0.0)

        for edge_name, prediction_edge in prediction.edges.items():
            if edge_name not in target.edges:
                raise ValueError(f"Missing edge '{edge_name}' in target graph.")
            target_edge = target.edges[edge_name]
            edge_sq_error_sum, edge_n_values = self._edge_gradient_and_mse(prediction_edge, target_edge)
            sq_error_sum = sq_error_sum + edge_sq_error_sum
            n_values = n_values + edge_n_values

        return sq_error_sum / jnp.maximum(n_values, 1.0)

    def get_context(self, get_info: bool = False) -> tuple[JaxGraph, dict[str, Any]]:
        info: dict[str, Any] = {}
        if get_info:
            info = {
                "batch_size": np.array(len(self.problems)),
                "current_shape": deepcopy(self.jax_context_batch.current_shape),
                "true_shape": deepcopy(self.jax_context_batch.true_shape),
            }
        return self.jax_context_batch, info

    def get_gradient(self, *, decision: JaxGraph, get_info: bool = False) -> tuple[JaxGraph, dict[str, Any]]:
        gradient = self._batch_grad_fn(decision)
        info: dict[str, Any] = {}
        if get_info:
            mse = self.loss(prediction=decision)
            info = {"mse": mse, "rmse": jnp.sqrt(mse)}
        return gradient, info

    def get_metrics(self, *, decision: JaxGraph, get_info: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
        mse = self.loss(prediction=decision)
        rmse = jnp.sqrt(mse)
        metrics = np.atleast_1d(np.asarray(rmse, dtype=np.float32))
        info: dict[str, Any] = {}
        if get_info:
            info = {"mse": mse, "rmse": rmse, "batch_size": np.array(len(self.problems))}
        return metrics, info

    @classmethod
    def from_dataset(
        cls,
        dataset: list[Any],
        context_max_shape: GraphShape | None = None,
        oracle_max_shape: GraphShape | None = None,
    ) -> "LoadFlowBatch":
        problems = [sample if isinstance(sample, LoadFlowProblem) else LoadFlowProblem(net=sample) for sample in dataset]
        return cls(problems=problems, context_max_shape=context_max_shape, oracle_max_shape=oracle_max_shape)


class LoadFlowDataLoader(ProblemLoader):
    def __init__(self, dataset: list[Any], batch_size: int, shuffle: bool = False, seed: int | None = None):
        if batch_size <= 0:
            raise ValueError("batch_size must be strictly positive.")

        self.dataset = list(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._current_idx = 0
        self._order = np.arange(len(self.dataset), dtype=np.int32)

    def __iter__(self) -> Iterator[ProblemBatch]:
        self._current_idx = 0
        self._order = np.arange(len(self.dataset), dtype=np.int32)
        if self.shuffle and len(self._order) > 1:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self._order)
        return self

    def __next__(self) -> ProblemBatch:
        if self._current_idx >= len(self.dataset):
            raise StopIteration

        end = min(self._current_idx + self.batch_size, len(self.dataset))
        idx_slice = self._order[self._current_idx : end]
        self._current_idx = end

        dataset_slice = [self.dataset[int(i)] for i in idx_slice]
        return LoadFlowBatch.from_dataset(dataset_slice)

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    @property
    def context_structure(self) -> GraphStructure:
        return LoadFlowProblem.CONTEXT_STRUCTURE

    @property
    def decision_structure(self) -> GraphStructure:
        return LoadFlowProblem.DECISION_STRUCTURE


def generate_3bus_problem(seed: int | None = None):
    """
        Randomized three-bus demo for supervised load-flow learning.

        Context graph:
            - lines: network edges between buses
            - generators: slack/PV units with partial state masks
            - loads: PQ demands

        Oracle graph:
            - lines: line flows
            - generators: solved generator states
            - loads: solved load bus states

    Conventions:
            line.kind: 0=line, 1=trafo
            generator_type: 0=PV, 1=slack
            Unknown context variables are encoded as 0.0 plus *_mask=0.
    """

    rng = np.random.default_rng(seed)
    S_BASE_MVA = 100.0

    n_addresses = 3
    registry = np.arange(n_addresses, dtype=np.int32)

    line_from_bus = np.array([0, 1, 0], dtype=np.int32)
    line_to_bus = np.array([1, 2, 2], dtype=np.int32)
    line_kind = np.array([0, 0, 1], dtype=np.int32)

    base_r = rng.uniform(0.0015, 0.0120, size=3).astype(np.float32)
    base_x = rng.uniform(0.0100, 0.1200, size=3).astype(np.float32)
    base_b = rng.uniform(0.0000, 0.0030, size=3).astype(np.float32)
    tap = np.array([1.0, 1.0, float(rng.uniform(0.95, 1.05))], dtype=np.float32)
    phase = np.array([0.0, 0.0, float(rng.uniform(-4.0, 4.0))], dtype=np.float32)

    p_load_mw = float(rng.uniform(450.0, 950.0))
    q_load_mvar = float(rng.uniform(120.0, 360.0))

    p_pv_mw = float(rng.uniform(0.35 * p_load_mw, 0.85 * p_load_mw))
    p_slack_mw = p_load_mw - p_pv_mw

    q_pv_mvar = float(rng.uniform(-120.0, 220.0))
    q_slack_mvar = q_load_mvar - q_pv_mvar

    p0 = np.float32(p_slack_mw / S_BASE_MVA)
    p1 = np.float32(-p_load_mw / S_BASE_MVA)
    p2 = np.float32(p_pv_mw / S_BASE_MVA)
    q0 = np.float32(q_slack_mvar / S_BASE_MVA)
    q1 = np.float32(-q_load_mvar / S_BASE_MVA)
    q2 = np.float32(q_pv_mvar / S_BASE_MVA)

    p_aux = np.float32(rng.uniform(-0.5, 0.5) * max(abs(float(p0)), 0.1))
    q_aux = np.float32(rng.uniform(-0.5, 0.5) * max(abs(float(q0)), 0.1))

    p02 = p_aux
    p01 = p0 - p02
    p12 = p1 + p01

    q02 = q_aux
    q01 = q0 - q02
    q12 = q1 + q01

    p_from = np.array([p01, p12, p02], dtype=np.float32)
    p_to = -p_from
    q_from = np.array([q01, q12, q02], dtype=np.float32)
    q_to = -q_from

    rating = (1.5 * np.maximum(np.abs(p_from), np.abs(q_from)) + 0.5).astype(np.float32)

    line_features = {
        "kind": line_kind.astype(np.float32),
        "r_pu": base_r,
        "x_pu": base_x,
        "b_pu": base_b,
        "tap": tap,
        "phase_shift_deg": phase,
        "rating_pu": rating,
    }
    line_edge = Edge.from_dict(address_dict={"from_bus": line_from_bus, "to_bus": line_to_bus}, feature_dict=line_features)

    gen_bus = np.array([0, 2], dtype=np.int32)
    gen_type = np.array([1.0, 0.0], dtype=np.float32)
    gen_p_mw = np.array([0.0, p_pv_mw], dtype=np.float32)
    gen_q_mvar = np.array([0.0, 0.0], dtype=np.float32)
    gen_vm_set = np.array([1.00, float(rng.uniform(0.99, 1.04))], dtype=np.float32)
    gen_va_set = np.array([0.0, 0.0], dtype=np.float32)

    generator_features = {
        "generator_type": gen_type,
        "p_set_pu": gen_p_mw / S_BASE_MVA,
        "q_set_pu": gen_q_mvar / S_BASE_MVA,
        "vm_pu_set": gen_vm_set,
        "va_deg_set": gen_va_set,
        "vm_mask": np.array([1.0, 1.0], dtype=np.float32),
        "va_mask": np.array([1.0, 0.0], dtype=np.float32),
        "p_mask": np.array([0.0, 1.0], dtype=np.float32),
        "q_mask": np.array([0.0, 0.0], dtype=np.float32),
        "p_min_pu": np.array([0.0, 0.0], dtype=np.float32),
        "p_max_pu": np.array(
            [max(30.0, 1.4 * p_slack_mw / S_BASE_MVA), max(12.0, 1.2 * p_pv_mw / S_BASE_MVA)],
            dtype=np.float32,
        ),
        "q_min_pu": np.array([min(-15.0, 1.2 * q_slack_mvar / S_BASE_MVA), -4.0], dtype=np.float32),
        "q_max_pu": np.array([max(15.0, 1.2 * q_slack_mvar / S_BASE_MVA), 4.0], dtype=np.float32),
    }
    generator_edge = Edge.from_dict(address_dict={"bus": gen_bus}, feature_dict=generator_features)

    load_bus = np.array([1], dtype=np.int32)
    load_features = {
        "p_set_pu": np.array([p1], dtype=np.float32),
        "q_set_pu": np.array([q1], dtype=np.float32),
    }
    load_edge = Edge.from_dict(address_dict={"bus": load_bus}, feature_dict=load_features)

    line_state_features = {
        "p_from_pu": p_from,
        "q_from_pu": q_from,
        "p_to_pu": p_to,
        "q_to_pu": q_to,
    }
    line_state_edge = Edge.from_dict(
        address_dict={"from_bus": line_from_bus, "to_bus": line_to_bus},
        feature_dict=line_state_features,
    )

    vm_slack = np.float32(1.00)
    vm_pv = np.float32(gen_vm_set[1])
    vm_load = np.float32(np.clip(rng.normal(0.99, 0.015), 0.94, 1.05))

    va_slack = np.float32(0.0)
    va_pv = np.float32(rng.uniform(-4.0, 1.0))
    va_load = np.float32(rng.uniform(-10.0, -1.0))

    generator_state_features = {
        "p_pu": np.array([p0, p2], dtype=np.float32),
        "q_pu": np.array([q0, q2], dtype=np.float32),
        "vm_pu": np.array([vm_slack, vm_pv], dtype=np.float32),
        "va_deg": np.array([va_slack, va_pv], dtype=np.float32),
    }
    generator_state_edge = Edge.from_dict(address_dict={"bus": gen_bus}, feature_dict=generator_state_features)

    load_state_features = {
        "p_pu": np.array([p1], dtype=np.float32),
        "q_pu": np.array([q1], dtype=np.float32),
        "vm_pu": np.array([vm_load], dtype=np.float32),
        "va_deg": np.array([va_load], dtype=np.float32),
    }

    assert np.isclose(float(np.sum(generator_state_features["p_pu"]) + np.sum(load_state_features["p_pu"])), 0.0, atol=1e-6)
    assert np.isclose(float(np.sum(generator_state_features["q_pu"]) + np.sum(load_state_features["q_pu"])), 0.0, atol=1e-6)
    load_state_edge = Edge.from_dict(address_dict={"bus": load_bus}, feature_dict=load_state_features)

    edges = {
        "lines": line_edge,
        "generators": generator_edge,
        "loads": load_edge,
    }

    oracle_edges = {
        "lines": line_state_edge,
        "generators": generator_state_edge,
        "loads": load_state_edge,
    }

    context_graph = Graph.from_dict(edge_dict=edges, registry=registry)
    oracle_graph = Graph.from_dict(edge_dict=oracle_edges, registry=registry)

    return LoadFlowProblem(context=context_graph, oracle=oracle_graph)


def generate_3bus_batch(
    batch_size: int,
    seed: int | None = None,
    context_max_shape: GraphShape | None = None,
    oracle_max_shape: GraphShape | None = None,
) -> LoadFlowBatch:
    if batch_size <= 0:
        raise ValueError("batch_size must be strictly positive.")

    rng = np.random.default_rng(seed)
    problem_seeds = rng.integers(low=0, high=np.iinfo(np.int32).max, size=batch_size, dtype=np.int32)
    problems = [generate_3bus_problem(seed=int(s)) for s in problem_seeds]

    return LoadFlowBatch(
        problems=problems,
        context_max_shape=context_max_shape,
        oracle_max_shape=oracle_max_shape,
    )


def print_graph_summary(graph: Graph) -> None:
    print("Graph true shape:", graph.true_shape.to_jsonable_dict())
    print("Graph current shape:", graph.current_shape.to_jsonable_dict())
    print("Edges:")
    for edge_name, edge in sorted(graph.edges.items()):
        feature_names = [] if edge.feature_names is None else list(sorted(edge.feature_names.keys()))
        address_names = [] if edge.address_names is None else list(sorted(edge.address_names.keys()))
        print(f"  - {edge_name}: n_obj={edge.n_obj}, addresses={address_names}, features={feature_names}")


def summarize_shape_values(values: Any) -> str | int | float:
    arr = np.asarray(values)
    if arr.ndim == 0:
        scalar = float(arr)
        if np.isclose(scalar, round(scalar)):
            return int(round(scalar))
        return scalar

    flat = arr.reshape(-1)
    unique_vals, counts = np.unique(flat, return_counts=True)
    parts = []
    for val, count in zip(unique_vals, counts):
        v = float(val)
        if np.isclose(v, round(v)):
            val_txt = str(int(round(v)))
        else:
            val_txt = f"{v:.4g}"
        parts.append(f"{val_txt} x{int(count)}")
    return ", ".join(parts)


def shape_to_printable_dict(shape) -> dict[str, Any]:
    return {
        "edges": {k: summarize_shape_values(v) for k, v in shape.edges.items()},
        "addresses": summarize_shape_values(shape.addresses),
    }


def print_section(title: str, payload: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    for key, value in payload.items():
        print(f"- {key}: {value}")

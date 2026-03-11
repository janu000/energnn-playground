import numpy as np
import jax
import jax.numpy as jnp

from energnn.graph.edge import Edge
from energnn.graph.graph import Graph
from energnn.graph import EdgeStructure, GraphStructure
from energnn.graph.jax import JaxGraph, JaxEdge
from energnn.problem import Problem
from energnn.problem.metadata import ProblemMetadata


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
        context: Graph,
        oracle: Graph,
    ):
        self._name = "load-flow-problem"
        self._context = context
        self._oracle = oracle
        self.jax_context = JaxGraph.from_numpy_graph(context)
        self.jax_oracle = JaxGraph.from_numpy_graph(oracle)
        self._decision_grad_fn = jax.grad(lambda pred: self.loss(prediction=pred))

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
        # Framework compatibility: trainer passes `decision`, so gradient is d loss / d decision.
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

        # Trainer expects list/array-like metrics.
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
        raise NotImplementedError("Save not implemented for demo")

    @property
    def context_structure(self) -> GraphStructure:
        return self.CONTEXT_STRUCTURE

    @property
    def decision_structure(self) -> GraphStructure:
        return self.DECISION_STRUCTURE


def generate_3bus_problem():
    """
        Three-bus demo for supervised load-flow learning.

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

    S_BASE_MVA = 100.0

    # Registry / addresses (buses are node IDs)
    n_addresses = 3
    registry = np.arange(n_addresses, dtype=np.int32)

    # -----------------------------------------
    # lines edge (line + transformer combined)
    # -----------------------------------------
    line_from_bus = np.array([0, 1, 0], dtype=np.int32)
    line_to_bus = np.array([1, 2, 2], dtype=np.int32)
    line_kind = np.array([0, 0, 1], dtype=np.int32)  # 0=line, 1=trafo

    # Shared feature schema across all branches.
    line_features = {
        "kind": line_kind.astype(np.float32),
        "r_pu": np.array([0.0030, 0.0022, 0.0100], dtype=np.float32),
        "x_pu": np.array([0.0150, 0.0110, 0.1000], dtype=np.float32),
        "b_pu": np.array([0.0010, 0.0008, 0.0000], dtype=np.float32),
        "tap": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "phase_shift_deg": np.array([0.0, 0.0, 2.0], dtype=np.float32),
        "rating_pu": np.array([10.0, 8.0, 5.0], dtype=np.float32),
    }
    line_edge = Edge.from_dict(
        address_dict={"from_bus": line_from_bus, "to_bus": line_to_bus},
        feature_dict=line_features
    )

    # -----------------------------------------
    # generators edge: slack + PV
    # -----------------------------------------
    # bus 0: slack, bus 2: PV generator
    gen_bus = np.array([0, 2], dtype=np.int32)
    gen_type = np.array([1.0, 0.0], dtype=np.float32)  # 1=slack, 0=PV
    gen_p_mw = np.array([0.0, 800.0], dtype=np.float32)
    gen_q_mvar = np.array([0.0, 0.0], dtype=np.float32)
    gen_vm_set = np.array([1.00, 1.02], dtype=np.float32)
    gen_va_set = np.array([0.0, 0.0], dtype=np.float32)

    generator_features = {
        "generator_type": gen_type,
        "p_set_pu": gen_p_mw / S_BASE_MVA,
        "q_set_pu": gen_q_mvar / S_BASE_MVA,
        "vm_pu_set": gen_vm_set,
        "va_deg_set": gen_va_set,
        # Slack knows vm+va, PV knows vm+p.
        "vm_mask": np.array([1.0, 1.0], dtype=np.float32),
        "va_mask": np.array([1.0, 0.0], dtype=np.float32),
        "p_mask": np.array([0.0, 1.0], dtype=np.float32),
        "q_mask": np.array([0.0, 0.0], dtype=np.float32),
        "p_min_pu": np.array([0.0, 0.0], dtype=np.float32),
        "p_max_pu": np.array([30.0, 12.0], dtype=np.float32),
        "q_min_pu": np.array([-15.0, -4.0], dtype=np.float32),
        "q_max_pu": np.array([15.0, 4.0], dtype=np.float32),
    }
    generator_edge = Edge.from_dict(address_dict={"bus": gen_bus}, feature_dict=generator_features)

    # -----------------------------------------
    # loads edge: PQ demands
    # -----------------------------------------
    load_bus = np.array([1], dtype=np.int32)
    load_features = {
        "load_type": np.array([0.0], dtype=np.float32),
        "p_set_pu": np.array([-7.0], dtype=np.float32),
        "q_set_pu": np.array([-2.0], dtype=np.float32),
        "vm_pu_set": np.array([0.0], dtype=np.float32),
        "va_deg_set": np.array([0.0], dtype=np.float32),
        "vm_mask": np.array([0.0], dtype=np.float32),
        "va_mask": np.array([0.0], dtype=np.float32),
        "p_mask": np.array([1.0], dtype=np.float32),
        "q_mask": np.array([1.0], dtype=np.float32),
    }
    load_edge = Edge.from_dict(address_dict={"bus": load_bus}, feature_dict=load_features)

    # -----------------------------------------
    # oracle state edges (supervision target)
    # -----------------------------------------
    line_state_features = {
        "p_from_pu": np.array([2.2, -1.7, -0.5], dtype=np.float32),
        "q_from_pu": np.array([0.8, -0.4, -0.2], dtype=np.float32),
        "p_to_pu": np.array([-2.1, 1.6, 0.48], dtype=np.float32),
        "q_to_pu": np.array([-0.75, 0.35, 0.18], dtype=np.float32),
    }
    line_state_edge = Edge.from_dict(
        address_dict={"from_bus": line_from_bus, "to_bus": line_to_bus},
        feature_dict=line_state_features,
    )

    generator_state_features = {
        "p_pu": np.array([0.8, 8.0], dtype=np.float32),
        "q_pu": np.array([2.05, 0.45], dtype=np.float32),
        "vm_pu": np.array([1.00, 1.02], dtype=np.float32),
        "va_deg": np.array([0.0, -1.1], dtype=np.float32),
    }
    generator_state_edge = Edge.from_dict(
        address_dict={"bus": gen_bus},
        feature_dict=generator_state_features,
    )

    load_state_features = {
        "p_pu": np.array([-7.0], dtype=np.float32),
        "q_pu": np.array([-2.5], dtype=np.float32),
        "vm_pu": np.array([0.985], dtype=np.float32),
        "va_deg": np.array([-3.2], dtype=np.float32),
    }
    load_state_edge = Edge.from_dict(
        address_dict={"bus": load_bus},
        feature_dict=load_state_features,
    )

    # -------------------------
    # Build Graphs
    # -------------------------
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


def print_graph_summary(graph: Graph) -> None:
    print("Graph true shape:", graph.true_shape.to_jsonable_dict())
    print("Graph current shape:", graph.current_shape.to_jsonable_dict())
    print("Edges:")
    for edge_name, edge in sorted(graph.edges.items()):
        feature_names = [] if edge.feature_names is None else list(sorted(edge.feature_names.keys()))
        address_names = [] if edge.address_names is None else list(sorted(edge.address_names.keys()))
        print(f"  - {edge_name}: n_obj={edge.n_obj}, addresses={address_names}, features={feature_names}")


if __name__ == "__main__":

    tbp = generate_3bus_problem()
    jax_g, info = tbp.get_context(get_info=True)
    g = jax_g.to_numpy_graph()
    print("----- LoadFlowProblem context (Graph) -----")
    print(g)
    print("Info:", info)
    print_graph_summary(g)

    print("\n----- Round-trip NumPy -> JAX -> NumPy -----")
    jax_graph = JaxGraph.from_numpy_graph(g)
    g_back = jax_graph.to_numpy_graph()
    print("JAX flat feature shape:", tuple(jax_graph.feature_flat_array.shape))
    print("Round-trip shape:", g_back.true_shape.to_jsonable_dict())

    print("\n----- Problem metadata -----")
    print(tbp.get_metadata())

    print("\n----- Oracle target shape (full grid state) -----")
    print(tbp.jax_oracle.to_numpy_graph().true_shape.to_jsonable_dict())

    print("Context structure:", tbp.context_structure)
    print("Decision structure:", tbp.decision_structure)
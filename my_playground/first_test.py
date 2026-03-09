import numpy as np

from energnn.graph.edge import Edge
from energnn.graph.graph import Graph
from energnn.graph import EdgeStructure, GraphStructure
from energnn.graph.jax import JaxGraph
from energnn.problem import Problem
from energnn.problem.metadata import ProblemMetadata


class ThreeBusProblem(Problem):
    def __init__(self, context: Graph, oracle: Graph, context_structure: GraphStructure, decision_structure: GraphStructure):
        self._name = "three-bus-problem"
        self._context = context
        self._oracle = oracle
        self._context_structure = context_structure
        self._decision_structure = decision_structure
        self.jax_context = JaxGraph.from_numpy_graph(context)
        self.jax_oracle = JaxGraph.from_numpy_graph(oracle)

    def get_context(self, get_info: bool = False):
        # Return the stored context graph. Optionally return a small info dict.
        if get_info:
            info = {
                "n_addresses": int(self._context.true_shape.addresses),
                "edges": {k: int(v) for k, v in self._context.true_shape.edges.items()},
            }
            return self._context, info
        return self._context

    def get_gradient(self, *, decision, get_info: bool = False):
        raise NotImplementedError("Gradient not implemented for demo")

    def get_metrics(self, *, decision, get_info: bool = False):
        raise NotImplementedError("Metrics not implemented for demo")

    def get_metadata(self):
        return ProblemMetadata(name=self._name, config_id="threebus-demo", version="0.2")

    def save(self, *, path: str) -> None:
        raise NotImplementedError("Save not implemented for demo")

    @property
    def context_structure(self) -> GraphStructure:
        return self._context_structure

    @property
    def decision_structure(self) -> GraphStructure:
        return self._decision_structure


def create_three_bus_problem():
    """
    Three-bus demo using exactly:
      - bus_edge: bus/node features + bus_type
      - branch_edge: line + trafo (single edge type with kind)
      - injection_edge: gen + load + slack (single edge type with kind)

    Conventions:
      bus_type: 0=slack, 1=PV, 2=PQ
      branch.kind: 0=line, 1=trafo
      inj.kind: 0=gen, 1=load, 2=slack
      load is NEGATIVE injection (p_mw < 0, q_mvar < 0)
    """

    S_BASE_MVA = 100.0

    # Registry / addresses (buses are node IDs)
    n_addresses = 3
    registry = np.arange(n_addresses, dtype=np.int32)

    # -------------------------
    # bus_edge (node features)
    # -------------------------
    bus_addresses = {"bus": np.array([0, 1, 2], dtype=np.int32)}
    bus_features = {
        "bus_type": np.array([0, 2, 1], dtype=np.int32),          # bus0=slack, bus1=PQ, bus2=PV
        "vm_init_pu": np.array([1.00, 1.00, 1.02], dtype=np.float32),
        "va_init_deg": np.array([0.0, 0.0, 0.0], dtype=np.float32),
    }
    bus_edge = Edge.from_dict(
        address_dict=bus_addresses,
        feature_dict=bus_features
    )

    # -----------------------------------------
    # branch_edge (lines + transformer combined)
    # -----------------------------------------
    # Two lines: 0-1 and 1-2; One transformer: 0-2
    branch_from = np.array([0, 1, 0], dtype=np.int32)
    branch_to   = np.array([1, 2, 2], dtype=np.int32)
    branch_kind = np.array([0, 0, 1], dtype=np.int32)  # 0=line, 1=trafo

    # Shared feature schema across all branches
    # Lines: tap=1.0; rating_mva=0.0 (unused)
    # Trafo: r/x are placeholders here; tap/rating_mva set
    branch_features = {
        "kind": branch_kind,      
        "r_pu": np.array([0.0030, 0.0022, 0.0100], dtype=np.float32),
        "x_pu": np.array([0.0150, 0.0110, 0.1000], dtype=np.float32),
        "tap": np.array([1.0, 1.0, 1.0], dtype=np.float32),     # Always 1.0 for lines
        "rating_pu": np.array([10.0, 8.0, 5.0], dtype=np.float32),
    }
    branch_edge = Edge.from_dict(
        address_dict={"from_bus": branch_from, "to_bus": branch_to},
        feature_dict=branch_features
    )

    # -----------------------------------------
    # injection_edge (gen + load + slack)
    # -----------------------------------------
    # Slack at bus 0, Load at bus 1, Generator at bus 2 (PV bus)
    inj_bus  = np.array([0, 1, 2], dtype=np.int32)
    inj_kind = np.array([2, 1, 0], dtype=np.int32)  # 2=slack, 1=load, 0=gen

    # Load is negative injection
    inj_p_mw   = np.array([0.0,  -700.0, +800.0], dtype=np.float32)
    inj_q_mvar = np.array([0.0,  -200.0,   0.0], dtype=np.float32)

    # Convert injections to per-unit on S_BASE_MVA
    inj_p_pu = inj_p_mw / S_BASE_MVA
    inj_q_pu = inj_q_mvar / S_BASE_MVA


    # Setpoints: meaningful for slack/gen, unused for load (set to 0.0)
    inj_vm_pu_set  = np.array([1.00, 0.0, 1.02], dtype=np.float32)
    inj_va_deg_set = np.array([0.0,  0.0, 0.0], dtype=np.float32)  # slack reference angle

    # Limits: meaningful for slack/gen, unused for load (0.0)
    
    inj_p_min_mw = np.array([0.0, 0.0,   0.0], dtype=np.float32)
    inj_p_max_mw = np.array([3000.0, 0.0, 1200.0], dtype=np.float32)
    inj_q_min_mvar = np.array([-1500.0, 0.0, -400.0], dtype=np.float32)  
    inj_q_max_mvar = np.array([+1500.0, 0.0, +400.0], dtype=np.float32)
    
    inj_p_min_pu = inj_p_min_mw / S_BASE_MVA
    inj_p_max_pu = inj_p_max_mw / S_BASE_MVA
    inj_q_min_pu = inj_q_min_mvar / S_BASE_MVA
    inj_q_max_pu = inj_q_max_mvar / S_BASE_MVA


    injection_features = {
        "kind": inj_kind,

        # injections in pu
        "p_pu": inj_p_pu,
        "q_pu": inj_q_pu,

        # setpoints
        "vm_pu_set": inj_vm_pu_set,
        "va_deg_set": inj_va_deg_set,

        # limits in pu
        "p_min_pu": inj_p_min_pu,
        "p_max_pu": inj_p_max_pu,
        "q_min_pu": inj_q_min_pu,
        "q_max_pu": inj_q_max_pu,
    }

    injection_edge = Edge.from_dict(
        address_dict={"bus": inj_bus},
        feature_dict=injection_features
    )

    context_structure = GraphStructure(edges={
        "bus": EdgeStructure.from_list(address_list=["bus"], feature_list=list(bus_features.keys())),
        "branch": EdgeStructure.from_list(address_list=["from_bus", "to_bus"], feature_list=list(branch_features.keys())),
        "inj": EdgeStructure.from_list(address_list=["bus"], feature_list=list(injection_features.keys())),
    })

    decision_structure = GraphStructure(edges={
        "inj": EdgeStructure.from_list(address_list=["bus"], feature_list=["p_pu", "q_pu"]),
    })


    # -------------------------
    # Build Graphs
    # -------------------------
    edges = {
        "bus": bus_edge,
        "branch": branch_edge,
        "inj": injection_edge,
    }

    context_graph = Graph.from_dict(edge_dict=edges, registry=registry)
    oracle_graph = Graph.from_dict(edge_dict=edges, registry=registry)

    return ThreeBusProblem(context=context_graph, oracle=oracle_graph, context_structure=context_structure, decision_structure=decision_structure)


if __name__ == "__main__":
        
    tbp = create_three_bus_problem()
    g, info = tbp.get_context(get_info=True)
    print("----- ThreeBusProblem context (Graph) -----")
    print(g)
    print("Info:", info)
    print("Context structure:", tbp.context_structure)
    print("Decision structure:", tbp.decision_structure)
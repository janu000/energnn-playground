
**Owner:** @<[MCCS nextgen]\AIGriMod> 


[[_TOC_]]


# Introduction
---
_This page is a compilation of publicly available information from the EnerGNN public [repository](https://github.com/energnn/energnn) and official [documentation](https://energnn.readthedocs.io/en/latest) and does not claim any original authorship or development work._

**EnerGNN** is a Graph Neural Network library specifically designed for real-life energy networks, aiming to provide a practical GNN-oriented toolkit for learning and modeling on energy-infrastructure graphs rather than a generic graph ML framework. The project is supported by the institutions RTE, Université de Liège, and INRIA.


# Core Principles
---

"EnerGNN is based on [JAX](https://docs.jax.dev/en/latest/index.html) and [Flax](https://flax.readthedocs.io/en/latest/) and provides:
*   A **Hyper Heterogeneous Multi Graph** (H2MG) data representation, especially designed for large complex industrial networks (such as an Electrical Power Transmission System);
*   A compatible **Graph Neural Network** (GNN) library, robust to structure variations (outages, construction of new infrastructure, renaming / reordering, etc.);
*   A clear interface to help users apply **energnn** to their own custom use-cases."

![image.png](/.attachments/image-8eb28e64-0c1e-46ab-bfb8-b5c4577d7d8c.png)

In practice, EnerGNN combines three layers:

1. **Graph domain layer**: typed edge blocks with addresses and features.
2. **Model layer**: encoder/coupler/decoder style GNN components.
3. **Problem and trainer layer**: objective definition, batching, optimization, metrics, tracking, and checkpointing.


# Problem-Centric Design
---

EnerGNN training is built around a **Problem** abstraction.

A problem defines:

- **Context** (model input): structure and extraction logic.
- **Decision** (model output): structure and post-processing logic.
- **Oracle** (target/ground truth): supervision source.
- **Optimization interface**: gradient signal and metrics.

The key idea is that model training does not only depend on a network architecture, but on a domain problem that formalizes input, output, and objective behavior.


# Graph Representation
---

EnerGNN represents a network as a dictionary of edge blocks.

- A **Graph** is a mapping from edge-type name to an **Edge**:

```
edge_dict = {
    "bus": bus_edge,
    "line": line_edge
}
```

- Each **Edge** contains:
   - an `address_dict` (connectivity or port indices),
   - a `feature_dict` (typed feature tensors).

```
bus_addresses = {"bus": np.array([0, 1, 2], dtype=np.int32)}

bus_features = {
    "bus_type": np.array([0, 2, 1], dtype=np.int32), # bus0=slack, bus1=PQ, bus2=PV
    "vm_init_pu": np.array([1.00, 1.00, 1.02], dtype=np.float32),
    "va_init_deg": np.array([0.0, 0.0, 0.0], dtype=np.float32),
}

bus_edge = Edge.from_dict(
    address_dict=bus_addresses,
    feature_dict=bus_features
)
```

# Data Pipeline Concepts
---

The data pipeline is split into three responsibilities:

1. **Dataset / Problem instances**: map raw domain data into EnerGNN graph structures.
2. **Batch object**: collate multiple problem instances into batched tensors/graphs.
3. **Data loader**: iterate through dataset slices and yield batches.

### EnerGNN provides no domain-specific data importers out of the box. 
- Data ingestion must be implemented by the developer for each target source format
- Define source-specific parser(s) (e.g., CSV, CGMES).
- Map parsed records into EnerGNN `Graph` / `Edge` structures (`address_dict`, `feature_dict`).
- Integrate the importer into the project-specific dataset/problem pipeline.

# Model Composition in EnerGNN
--- 

EnerGNN models are generally modular and can be assembled from reusable blocks.

Common block roles:
- **Normalizer**: feature scaling/normalization for stable optimization.
- **Encoder**: project raw graph features into latent representations.
- **Coupler (Message Passing)**: iterative information exchange across graph structure.
- **Decoder**: map latent representations to decision-space predictions.





































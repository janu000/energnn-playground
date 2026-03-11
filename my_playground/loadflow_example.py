from energnn.graph.jax import JaxGraph
from pathlib import Path
from energnn.graph import visualize_graph

from loadflow_problem import LoadFlowDataLoader, LoadFlowProblem, print_section, shape_to_printable_dict


def _find_latest_dataset_path() -> Path:
    data_dir = Path(__file__).parent / "data"
    candidates = sorted(data_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No .pkl dataset found in {data_dir}")
    return candidates[0]


def main() -> None:
    dataset_path = _find_latest_dataset_path()
    dataset = LoadFlowProblem.load_dataset(dataset_path)
    if not dataset:
        raise ValueError(f"Dataset {dataset_path} does not contain any samples.")

    problem = LoadFlowProblem(net=dataset[0])
    context_jax, context_info = problem.get_context(get_info=True)
    context_np = context_jax.to_numpy_graph()

    print_section(
        "Single Problem",
        {
            "dataset_path": str(dataset_path),
            "dataset_size": len(dataset),
            "name": problem.get_metadata()["name"],
            "context_true_shape": context_np.true_shape.to_jsonable_dict(),
            "context_current_shape": context_np.current_shape.to_jsonable_dict(),
            "oracle_shape": problem.jax_oracle.to_numpy_graph().true_shape.to_jsonable_dict(),
            "context_info": context_info,
        },
    )

    preview_svg_path = Path(__file__).parent / "data" / "graph_preview.svg"
    try:
        visualize_graph(
            context_np,
            title="LoadFlow Single Graph",
            save_path=str(preview_svg_path),
            show=False,
        )
        print_section(
            "Visualization",
            {"graph_svg": str(preview_svg_path)},
        )
    except ImportError as exc:
        print_section("Visualization", {"status": str(exc)})

    jax_graph = JaxGraph.from_numpy_graph(context_np)
    roundtrip_np = jax_graph.to_numpy_graph()
    print_section(
        "Round Trip",
        {
            "jax_flat_feature_shape": tuple(jax_graph.feature_flat_array.shape),
            "roundtrip_true_shape": roundtrip_np.true_shape.to_jsonable_dict(),
        },
    )

    loader = LoadFlowDataLoader(dataset=dataset, batch_size=min(32, len(dataset)), shuffle=False)
    batch = next(iter(loader))
    batch_context, batch_info = batch.get_context(get_info=True)
    batch_context_np = batch_context.to_numpy_graph()
    print_section(
        "Batch Problem",
        {
            "batch_size": len(batch.problems),
            "batch_context_current_shape": shape_to_printable_dict(batch_context_np.current_shape),
            "batch_context_true_shape": shape_to_printable_dict(batch_context_np.true_shape),
            "batch_info_keys": sorted(batch_info.keys()),
        },
    )


if __name__ == "__main__":
    main()

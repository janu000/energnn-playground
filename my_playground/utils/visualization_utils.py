from pathlib import Path

import numpy as np


def visualize_graph(
    graph,
    *,
    title: str | None = None,
    edge_color_map: dict[str, str] | None = None,
    node_color: str = "#111111",
    save_path: str | None = None,
    show: bool = True,
):
    """
    Visualize a single EnerGNN graph as a static SVG figure.

    Edges with two or more address ports are drawn as links between node coordinates
    (first two sorted address keys are used). Unary edges are displayed as filled
    colored circles on the referenced node.

    :param graph: Single graph instance with ``edges`` and ``non_fictitious_addresses``.
    :param title: Optional plot title.
    :param edge_color_map: Optional mapping edge_type -> color.
    :param node_color: Default node color.
    :param save_path: Optional output SVG path. If None, writes ``graph_visualization.svg``.
    :param show: If True, call ``plt.show()``.
    :return: Tuple (fig, ax).
    :raises ImportError: If matplotlib is not installed.
    :raises ValueError: If the graph is batched.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "matplotlib and networkx are required for graph visualization. Install them with "
            "'uv add matplotlib networkx'."
        ) from exc

    if getattr(graph, "is_batch", False):
        raise ValueError("visualize_graph expects a single graph. Pass one sample, not a batch.")

    n_addresses = int(np.shape(graph.non_fictitious_addresses)[0])
    if n_addresses <= 0:
        raise ValueError("Graph has no addresses to visualize.")

    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(range(n_addresses))

    for _, edge in sorted(graph.edges.items()):
        if edge.non_fictitious is None:
            valid_mask = np.ones(edge.n_obj, dtype=bool)
        else:
            valid_mask = np.asarray(edge.non_fictitious).astype(bool)

        address_dict = edge.address_dict or {}
        port_names = sorted(address_dict.keys())
        if len(port_names) < 2:
            continue

        src = np.asarray(address_dict[port_names[0]], dtype=np.int64)
        dst = np.asarray(address_dict[port_names[1]], dtype=np.int64)
        for i in range(edge.n_obj):
            if not valid_mask[i]:
                continue
            a = int(src[i])
            b = int(dst[i])
            if a < 0 or b < 0 or a >= n_addresses or b >= n_addresses or a == b:
                continue
            layout_graph.add_edge(a, b)

    layout_iterations = 250 if n_addresses <= 200 else 120
    spring_k = 1.2 / np.sqrt(max(float(n_addresses), 1.0))
    pos = nx.spring_layout(
        layout_graph,
        seed=7,
        k=spring_k,
        iterations=layout_iterations,
        weight=None,
    )
    xy = np.array([pos[idx] for idx in range(n_addresses)], dtype=np.float32)

    max_abs = float(np.max(np.abs(xy))) if xy.size > 0 else 1.0
    scale = max(max_abs, 1e-6)
    xy = xy / scale

    fig, ax = plt.subplots(figsize=(7, 7))

    default_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    color_map = dict(edge_color_map or {})
    node_radius = float(np.clip(0.3 / np.sqrt(max(n_addresses, 1)), 0.02, 0.05))
    leaf_radius = 0.72 * node_radius
    leaf_offset = 2.4 * node_radius
    address_fontsize = float(np.clip(130.0 * node_radius, 1, 8))
    slack_leaf_positions: list[tuple[float, float]] = []
    bus_type_map = {0: "PQ", 1: "PV", 2: "Slack"}
    bus_type_color_map = {0: "#4b5563", 1: "#10b981", 2: "#ef4444"}
    bus_type_by_address = np.full(n_addresses, -1, dtype=np.int32)
    bus_color_by_address: dict[int, str] = {idx: node_color for idx in range(n_addresses)}
    line_edge_types: set[str] = set()
    unary_edge_types: dict[str, str] = {}
    generator_leaf_color: str | None = None
    leaf_specs_by_bus: dict[int, list[tuple[str, str, bool]]] = {idx: [] for idx in range(n_addresses)}

    buses_edge = graph.edges.get("buses") if hasattr(graph, "edges") else None
    if buses_edge is not None and buses_edge.feature_dict is not None:
        feature_dict = buses_edge.feature_dict
        if "bus_type" in feature_dict:
            raw_bus_type = np.asarray(feature_dict["bus_type"]).reshape(-1)
            for idx in range(min(n_addresses, raw_bus_type.shape[0])):
                t = int(round(float(raw_bus_type[idx])))
                bus_type_by_address[idx] = t
                if t in bus_type_color_map:
                    bus_color_by_address[idx] = bus_type_color_map[t]

    for edge_idx, (edge_name, edge) in enumerate(sorted(graph.edges.items())):
        color = color_map.get(edge_name, default_palette[edge_idx % len(default_palette)])
        color_map[edge_name] = color

        if edge.non_fictitious is None:
            valid_mask = np.ones(edge.n_obj, dtype=bool)
        else:
            valid_mask = np.asarray(edge.non_fictitious).astype(bool)

        address_dict = edge.address_dict or {}
        port_names = sorted(address_dict.keys())

        if len(port_names) >= 2:
            line_edge_types.add(edge_name)
            src = np.asarray(address_dict[port_names[0]], dtype=np.int64)
            dst = np.asarray(address_dict[port_names[1]], dtype=np.int64)

            for i in range(edge.n_obj):
                if not valid_mask[i]:
                    continue
                a = int(src[i])
                b = int(dst[i])
                if a < 0 or b < 0 or a >= n_addresses or b >= n_addresses:
                    continue
                x1, y1 = xy[a]
                x2, y2 = xy[b]
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=2.0, alpha=0.9, zorder=1)
        elif len(port_names) == 1:
            if edge_name == "buses":
                continue
            unary_edge_types[edge_name] = color
            nodes = np.asarray(address_dict[port_names[0]], dtype=np.int64)
            feature_dict = edge.feature_dict or {}
            generator_type = feature_dict.get("generator_type")
            for i in range(edge.n_obj):
                if not valid_mask[i]:
                    continue
                node = int(nodes[i])
                if node < 0 or node >= n_addresses:
                    continue
                is_slack = False
                if generator_type is not None and "generator" in edge_name.lower():
                    generator_leaf_color = color
                    if float(np.asarray(generator_type)[i]) >= 0.5:
                        is_slack = True
                leaf_specs_by_bus[node].append((edge_name, color, is_slack))

    center_x = float(np.mean(xy[:, 0])) if n_addresses > 0 else 0.0
    center_y = float(np.mean(xy[:, 1])) if n_addresses > 0 else 0.0

    for idx in range(n_addresses):
        node_patch = Circle(
            (xy[idx][0], xy[idx][1]),
            radius=node_radius,
            fill=True,
            linewidth=1.0,
            edgecolor="white",
            facecolor=bus_color_by_address[idx],
            zorder=2,
        )
        ax.add_patch(node_patch)

        leaf_specs = leaf_specs_by_bus[idx]
        if len(leaf_specs) == 0:
            continue

        base_angle = float(np.arctan2(float(xy[idx][1]) - center_y, float(xy[idx][0]) - center_x))
        if len(leaf_specs) == 1:
            leaf_angles = [base_angle]
        else:
            spread = np.pi / 2.0
            leaf_angles = np.linspace(base_angle - spread / 2.0, base_angle + spread / 2.0, len(leaf_specs))

        for leaf_idx, (_, leaf_color, is_slack) in enumerate(leaf_specs):
            angle = float(leaf_angles[leaf_idx])
            cx = float(xy[idx][0] + leaf_offset * np.cos(angle))
            cy = float(xy[idx][1] + leaf_offset * np.sin(angle))

            ax.plot(
                [xy[idx][0], cx],
                [xy[idx][1], cy],
                color=leaf_color,
                linewidth=1.2,
                alpha=0.9,
                zorder=1.4,
            )
            leaf_patch = Circle(
                (cx, cy),
                radius=leaf_radius,
                fill=True,
                linewidth=1.0,
                edgecolor="white",
                facecolor=leaf_color,
                zorder=2.2,
            )
            ax.add_patch(leaf_patch)
            if is_slack:
                slack_leaf_positions.append((cx, cy))

    for sx, sy in slack_leaf_positions:
        slack_ring = Circle(
            (sx, sy),
            radius=leaf_radius,
            fill=False,
            linewidth=2.2,
            edgecolor="#f59e0b",
            zorder=2.4,
        )
        ax.add_patch(slack_ring)

    for idx, (x, y) in enumerate(xy):
        ax.text(
            x,
            y,
            str(idx),
            ha="center",
            va="center",
            fontsize=address_fontsize,
            fontweight="bold",
            color="white",
            zorder=3,
        )

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)

    legend_handles = []
    if np.any(bus_type_by_address >= 0):
        present_types = sorted({int(x) for x in bus_type_by_address if int(x) in bus_type_map})
        for t in present_types:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    markersize=10,
                    markerfacecolor=bus_type_color_map[t],
                    markeredgecolor="white",
                    linestyle="None",
                    label=f"bus ({bus_type_map[t]})",
                )
            )
    else:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                markersize=10,
                markerfacecolor=node_color,
                markeredgecolor="white",
                linestyle="None",
                label="bus",
            )
        )
    for edge_name in sorted(line_edge_types):
        color = color_map[edge_name]
        legend_handles.append(Line2D([0], [0], color=color, lw=2.0, label=edge_name))
    for edge_name in sorted(unary_edge_types):
        color = unary_edge_types[edge_name]
        if edge_name == "generators":
            legend_label = "generator"
        elif edge_name == "loads":
            legend_label = "load"
        else:
            legend_label = edge_name
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                markersize=10,
                markerfacecolor=color,
                markeredgecolor="white",
                linestyle="None",
                label=legend_label,
            )
        )

    if slack_leaf_positions:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                markersize=10,
                markerfacecolor=generator_leaf_color or node_color,
                markeredgecolor="#f59e0b",
                markeredgewidth=2.2,
                linestyle="None",
                label="slack generator",
            )
        )

    if legend_handles:
        ax.legend(handles=legend_handles, loc="best", frameon=True, title="Legend")

    output_path = Path(save_path) if save_path is not None else Path("graph_visualization.svg")
    if output_path.suffix.lower() != ".svg":
        output_path = output_path.with_suffix(".svg")
    fig.savefig(str(output_path), format="svg", dpi=150, bbox_inches="tight", facecolor="white", transparent=False)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax

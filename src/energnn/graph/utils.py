# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
#
from pathlib import Path

import jax

# import jax.numpy as jnp
import numpy as np
from typing import Any


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
    Visualize a single EnerGNN graph with one color per edge type and a legend.

    Edges with two or more address ports are drawn as segments between node coordinates
    (first two sorted address keys are used). Unary edges are displayed as filled
    colored circles on the referenced node.

    :param graph: Single graph instance with ``edges`` and ``non_fictitious_addresses``.
    :param title: Optional plot title.
    :param edge_color_map: Optional mapping edge_type -> matplotlib color.
    :param node_color: Node color.
    :param save_path: Optional output image path.
    :param show: If True, call ``plt.show()``.
    :return: Tuple (fig, ax).
    :raises ImportError: If matplotlib is not installed.
    :raises ValueError: If the graph is batched.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle
    except ImportError as exc:
        raise ImportError("matplotlib is required for graph visualization. Install it with 'uv add matplotlib'.") from exc

    if getattr(graph, "is_batch", False):
        raise ValueError("visualize_graph expects a single graph. Pass one sample, not a batch.")

    n_addresses = int(np.shape(graph.non_fictitious_addresses)[0])
    if n_addresses <= 0:
        raise ValueError("Graph has no addresses to visualize.")

    angles = np.linspace(0.0, 2.0 * np.pi, n_addresses, endpoint=False)
    xy = np.stack([np.cos(angles), np.sin(angles)], axis=1)

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

    legend_handles = []
    color_map = dict(edge_color_map or {})

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
            nodes = np.asarray(address_dict[port_names[0]], dtype=np.int64)
            for i in range(edge.n_obj):
                if not valid_mask[i]:
                    continue
                node = int(nodes[i])
                if node < 0 or node >= n_addresses:
                    continue
                x, y = xy[node]
                node_fill = Circle(
                    (x, y),
                    radius=0.09,
                    fill=True,
                    linewidth=1.0,
                    edgecolor="white",
                    facecolor=color,
                    zorder=2,
                )
                ax.add_patch(node_fill)

        legend_handles.append(Line2D([0], [0], color=color, lw=2, label=edge_name))

    ax.scatter(xy[:, 0], xy[:, 1], s=90, c=node_color, zorder=0)
    for i, (x, y) in enumerate(xy):
        ax.text(x, y, str(i), ha="center", va="center", fontsize=10, fontweight="bold")

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    if legend_handles:
        ax.legend(handles=legend_handles, title="Edge Types", loc="upper right")

    if save_path is not None:
        save_format = Path(save_path).suffix.lower().lstrip(".") or "png"
        fig.savefig(
            save_path,
            format=save_format,
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def to_numpy(a: dict | np.ndarray | jax.Array | tuple | None) -> dict | np.ndarray | None:
    """
    Converts a NumPy array, JAX array, or tuple of values into a NumPy array (dtype float32),
    or converts the values in a dictionary accordingly.

    - If `a` is None, returns None.
    - If `a` is a np.ndarray, jax.Array, jnp.ndarray, or tuple, it is converted to a np.ndarray (float32).
    - If `a` is a dict with some values being arrays or tuples, only those values are converted;
      others remain unchanged.
    - In all other cases, a TypeError is raised.

    :param a: A np.ndarray, jax.Array, tuple, dict, or None.
    :returns: Either None, a np.ndarray, or a dict with the same keys and converted np.ndarray values.
    :raises TypeError: If `a` is not of an expected or supported type.
    """

    if a is None:
        return None

    def _to_np(x: Any) -> Any:
        # On traite np.ndarray, jax.Array et tuple
        if isinstance(x, (np.ndarray, jax.Array, np.ndarray, tuple)):
            return np.array(x, dtype=np.dtype("float32"))
        else:
            return x

    if isinstance(a, dict):
        output: dict[Any, np.array] = {}
        for key, value in a.items():
            output[key] = _to_np(value)  # seules les values “ArrayLike” seront converties
        return output

    # Cas array-like, tuple et object
    if isinstance(a, (np.ndarray, jax.Array, np.ndarray, tuple, object)):
        return _to_np(a)

    # elif isinstance(a, object): # A supprimer, car cette condition est toujours vraie
    #    return a

    raise TypeError(f"Type {type(a)} non pris en charge par to_numpy")


# def to_jax_numpy(a: dict | np.ndarray | jax.Array | tuple | None) -> dict | jax.Array | None:
#     """Converts a dictionary of numpy values into a dictionary of jax.numpy objects."""
#     if a is None:
#         return None
#     elif isinstance(a, np.ndarray) or isinstance(a, jax.Array) or isinstance(a, tuple):
#         return jnp.array(a, dtype=jnp.dtype("float32"))
#     elif isinstance(a, dict):
#         output_dict = dict()
#         for key, value in a.items():
#             if isinstance(value, np.ndarray) or isinstance(value, jax.Array) or isinstance(value, tuple):
#                 output_dict[key] = jnp.array(value, dtype=jnp.dtype("float32"))
#             else:
#                 output_dict[key] = value
#         return output_dict
#     elif isinstance(a, object):
#         return a
#     else:
#         raise TypeError()

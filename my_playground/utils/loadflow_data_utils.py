from __future__ import annotations

import numpy as np

from energnn.graph.edge import Edge
from energnn.graph.graph import Graph


def load_problem_from_pandapower_net(net, problem_cls):
    base_mva = float(getattr(net, "sn_mva", 100.0))
    bus_index = list(net.bus.index)
    bus_map = {int(b): i for i, b in enumerate(bus_index)}
    n_addresses = len(bus_index)
    registry = np.arange(n_addresses, dtype=np.int32)

    line_from, line_to, line_kind = [], [], []
    line_r, line_x, line_b, line_tap, line_phase, line_rating = [], [], [], [], [], []

    if hasattr(net, "line") and len(net.line) > 0:
        for _, row in net.line.iterrows():
            fb = bus_map[int(row.from_bus)]
            tb = bus_map[int(row.to_bus)]
            vn_kv = float(net.bus.loc[row.from_bus, "vn_kv"])
            z_base = (vn_kv**2) / base_mva
            r_ohm = float(row.r_ohm_per_km) * float(row.length_km)
            x_ohm = float(row.x_ohm_per_km) * float(row.length_km)

            line_from.append(fb)
            line_to.append(tb)
            line_kind.append(0.0)
            line_r.append(r_ohm / z_base)
            line_x.append(x_ohm / z_base)
            line_b.append(0.0)
            line_tap.append(1.0)
            line_phase.append(0.0)

            if "max_i_ka" in row and not np.isnan(row.max_i_ka):
                s_max_mva = np.sqrt(3.0) * vn_kv * float(row.max_i_ka)
                line_rating.append(s_max_mva / base_mva)
            else:
                line_rating.append(2.0)

    if hasattr(net, "trafo") and len(net.trafo) > 0:
        for _, row in net.trafo.iterrows():
            fb = bus_map[int(row.hv_bus)]
            tb = bus_map[int(row.lv_bus)]
            sn_mva = float(row.sn_mva)
            r_pu_traf = float(row.vkr_percent) / 100.0
            z_pu_traf = float(row.vk_percent) / 100.0
            x_pu_traf = float(np.sqrt(max(z_pu_traf**2 - r_pu_traf**2, 0.0)))
            scale = base_mva / sn_mva

            tap_pos = float(row.tap_pos) if "tap_pos" in row and not np.isnan(row.tap_pos) else 0.0
            tap_neutral = float(row.tap_neutral) if "tap_neutral" in row and not np.isnan(row.tap_neutral) else 0.0
            tap_step_pct = (
                float(row.tap_step_percent) if "tap_step_percent" in row and not np.isnan(row.tap_step_percent) else 0.0
            )
            tap = 1.0 + (tap_pos - tap_neutral) * tap_step_pct / 100.0
            shift_deg = float(row.shift_degree) if "shift_degree" in row and not np.isnan(row.shift_degree) else 0.0

            line_from.append(fb)
            line_to.append(tb)
            line_kind.append(1.0)
            line_r.append(r_pu_traf * scale)
            line_x.append(x_pu_traf * scale)
            line_b.append(0.0)
            line_tap.append(tap)
            line_phase.append(shift_deg)
            line_rating.append(sn_mva / base_mva)

    line_edge = Edge.from_dict(
        address_dict={"from_bus": np.asarray(line_from, dtype=np.int32), "to_bus": np.asarray(line_to, dtype=np.int32)},
        feature_dict={
            "kind": np.asarray(line_kind, dtype=np.float32),
            "r_pu": np.asarray(line_r, dtype=np.float32),
            "x_pu": np.asarray(line_x, dtype=np.float32),
            "b_pu": np.asarray(line_b, dtype=np.float32),
            "tap": np.asarray(line_tap, dtype=np.float32),
            "phase_shift_deg": np.asarray(line_phase, dtype=np.float32),
            "rating_pu": np.asarray(line_rating, dtype=np.float32),
        },
    )

    bus_type = np.zeros(n_addresses, dtype=np.float32)
    p_load_pu = np.zeros(n_addresses, dtype=np.float32)
    q_load_pu = np.zeros(n_addresses, dtype=np.float32)
    p_gen_pu = np.zeros(n_addresses, dtype=np.float32)
    q_gen_set_pu = np.zeros(n_addresses, dtype=np.float32)
    vm_set_pu = np.zeros(n_addresses, dtype=np.float32)
    va_set_deg = np.zeros(n_addresses, dtype=np.float32)
    p_gen_min_pu = np.zeros(n_addresses, dtype=np.float32)
    p_gen_max_pu = np.zeros(n_addresses, dtype=np.float32)
    q_gen_min_pu = np.zeros(n_addresses, dtype=np.float32)
    q_gen_max_pu = np.zeros(n_addresses, dtype=np.float32)

    vm_pu = np.zeros(n_addresses, dtype=np.float32)
    va_deg = np.zeros(n_addresses, dtype=np.float32)
    q_gen_pu = np.zeros(n_addresses, dtype=np.float32)

    if hasattr(net, "res_bus") and len(net.res_bus) > 0:
        for bus_orig, bus_idx in bus_map.items():
            vm_pu[bus_idx] = float(net.res_bus.loc[bus_orig, "vm_pu"])
            va_deg[bus_idx] = float(net.res_bus.loc[bus_orig, "va_degree"])

    if hasattr(net, "ext_grid") and len(net.ext_grid) > 0:
        res_ext = getattr(net, "res_ext_grid", None)
        for i, row in net.ext_grid.iterrows():
            b = bus_map[int(row.bus)]
            bus_type[b] = 2.0
            vm_set_pu[b] = float(row.vm_pu)
            va_set_deg[b] = float(row.va_degree) if "va_degree" in row and not np.isnan(row.va_degree) else 0.0
            p_gen_min_pu[b] = 0.0
            p_gen_max_pu[b] = 30.0
            q_gen_min_pu[b] = -15.0
            q_gen_max_pu[b] = 15.0

            p_res = float(res_ext.loc[i, "p_mw"]) if res_ext is not None and len(res_ext) > 0 else 0.0
            q_res = float(res_ext.loc[i, "q_mvar"]) if res_ext is not None and len(res_ext) > 0 else 0.0
            p_gen_pu[b] += p_res / base_mva
            q_gen_pu[b] += q_res / base_mva

    if hasattr(net, "gen") and len(net.gen) > 0:
        res_gen = getattr(net, "res_gen", None)
        for i, row in net.gen.iterrows():
            b = bus_map[int(row.bus)]
            if bus_type[b] != 2.0:
                bus_type[b] = 1.0
            p_gen_pu[b] += float(row.p_mw) / base_mva
            vm_set_pu[b] = float(row.vm_pu)

            pmin = float(row.min_p_mw) / base_mva if "min_p_mw" in row and not np.isnan(row.min_p_mw) else 0.0
            pmax = float(row.max_p_mw) / base_mva if "max_p_mw" in row and not np.isnan(row.max_p_mw) else 12.0
            qmin = float(row.min_q_mvar) / base_mva if "min_q_mvar" in row and not np.isnan(row.min_q_mvar) else -4.0
            qmax = float(row.max_q_mvar) / base_mva if "max_q_mvar" in row and not np.isnan(row.max_q_mvar) else 4.0
            p_gen_min_pu[b] += pmin
            p_gen_max_pu[b] += pmax
            q_gen_min_pu[b] += qmin
            q_gen_max_pu[b] += qmax

            p_res = float(res_gen.loc[i, "p_mw"]) if res_gen is not None and len(res_gen) > 0 else float(row.p_mw)
            q_res = float(res_gen.loc[i, "q_mvar"]) if res_gen is not None and len(res_gen) > 0 else 0.0
            q_gen_pu[b] += q_res / base_mva

    load_bus = []
    if hasattr(net, "load") and len(net.load) > 0:
        for _, row in net.load.iterrows():
            b = bus_map[int(row.bus)]
            load_bus.append(b)
            p_load_pu[b] += float(row.p_mw) / base_mva
            q_load_pu[b] += float(row.q_mvar) / base_mva

    bus_edge = Edge.from_dict(
        address_dict=None,
        feature_dict={
            "bus_type": bus_type,
            "p_load_pu": p_load_pu,
            "q_load_pu": q_load_pu,
            "p_gen_pu": p_gen_pu,
            "q_gen_set_pu": q_gen_set_pu,
            "vm_set_pu": vm_set_pu,
            "va_set_deg": va_set_deg,
            "p_gen_min_pu": p_gen_min_pu,
            "p_gen_max_pu": p_gen_max_pu,
            "q_gen_min_pu": q_gen_min_pu,
            "q_gen_max_pu": q_gen_max_pu,
        },
    )

    bus_state_edge = Edge.from_dict(
        address_dict=None,
        feature_dict={
            "vm_pu": vm_pu,
            "va_deg": va_deg,
            "q_gen_pu": q_gen_pu,
        },
    )

    context_graph = Graph.from_dict(
        edge_dict={"lines": line_edge, "buses": bus_edge},
        registry=registry,
    )
    oracle_graph = Graph.from_dict(
        edge_dict={"buses": bus_state_edge},
        registry=registry,
    )
    return problem_cls(context=context_graph, oracle=oracle_graph)

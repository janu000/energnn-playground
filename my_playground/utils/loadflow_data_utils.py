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

    gen_bus, gen_type = [], []
    p_set, q_set, vm_set, va_set = [], [], [], []
    vm_mask, va_mask, p_mask, q_mask = [], [], [], []
    p_min, p_max, q_min, q_max = [], [], [], []
    g_p, g_q, g_vm, g_va = [], [], [], []

    if hasattr(net, "ext_grid") and len(net.ext_grid) > 0:
        res_ext = getattr(net, "res_ext_grid", None)
        for i, row in net.ext_grid.iterrows():
            b = int(row.bus)
            gen_bus.append(bus_map[b])
            gen_type.append(1.0)
            p_set.append(0.0)
            q_set.append(0.0)
            vm_set.append(float(row.vm_pu))
            va_set.append(float(row.va_degree) if "va_degree" in row and not np.isnan(row.va_degree) else 0.0)
            vm_mask.append(1.0)
            va_mask.append(1.0)
            p_mask.append(0.0)
            q_mask.append(0.0)
            p_min.append(0.0)
            p_max.append(30.0)
            q_min.append(-15.0)
            q_max.append(15.0)

            p_res = float(res_ext.loc[i, "p_mw"]) if res_ext is not None and len(res_ext) > 0 else 0.0
            q_res = float(res_ext.loc[i, "q_mvar"]) if res_ext is not None and len(res_ext) > 0 else 0.0
            g_p.append(p_res / base_mva)
            g_q.append(q_res / base_mva)
            g_vm.append(float(net.res_bus.loc[b, "vm_pu"]))
            g_va.append(float(net.res_bus.loc[b, "va_degree"]))

    if hasattr(net, "gen") and len(net.gen) > 0:
        res_gen = getattr(net, "res_gen", None)
        for i, row in net.gen.iterrows():
            b = int(row.bus)
            gen_bus.append(bus_map[b])
            gen_type.append(0.0)
            p_set.append(float(row.p_mw) / base_mva)
            q_set.append(0.0)
            vm_set.append(float(row.vm_pu))
            va_set.append(0.0)
            vm_mask.append(1.0)
            va_mask.append(0.0)
            p_mask.append(1.0)
            q_mask.append(0.0)

            pmin = float(row.min_p_mw) / base_mva if "min_p_mw" in row and not np.isnan(row.min_p_mw) else 0.0
            pmax = float(row.max_p_mw) / base_mva if "max_p_mw" in row and not np.isnan(row.max_p_mw) else 12.0
            qmin = float(row.min_q_mvar) / base_mva if "min_q_mvar" in row and not np.isnan(row.min_q_mvar) else -4.0
            qmax = float(row.max_q_mvar) / base_mva if "max_q_mvar" in row and not np.isnan(row.max_q_mvar) else 4.0
            p_min.append(pmin)
            p_max.append(pmax)
            q_min.append(qmin)
            q_max.append(qmax)

            p_res = float(res_gen.loc[i, "p_mw"]) if res_gen is not None and len(res_gen) > 0 else float(row.p_mw)
            q_res = float(res_gen.loc[i, "q_mvar"]) if res_gen is not None and len(res_gen) > 0 else 0.0
            g_p.append(p_res / base_mva)
            g_q.append(q_res / base_mva)
            g_vm.append(float(net.res_bus.loc[b, "vm_pu"]))
            g_va.append(float(net.res_bus.loc[b, "va_degree"]))

    generator_edge = Edge.from_dict(
        address_dict={"bus": np.asarray(gen_bus, dtype=np.int32)},
        feature_dict={
            "generator_type": np.asarray(gen_type, dtype=np.float32),
            "p_set_pu": np.asarray(p_set, dtype=np.float32),
            "q_set_pu": np.asarray(q_set, dtype=np.float32),
            "vm_pu_set": np.asarray(vm_set, dtype=np.float32),
            "va_deg_set": np.asarray(va_set, dtype=np.float32),
            "vm_mask": np.asarray(vm_mask, dtype=np.float32),
            "va_mask": np.asarray(va_mask, dtype=np.float32),
            "p_mask": np.asarray(p_mask, dtype=np.float32),
            "q_mask": np.asarray(q_mask, dtype=np.float32),
            "p_min_pu": np.asarray(p_min, dtype=np.float32),
            "p_max_pu": np.asarray(p_max, dtype=np.float32),
            "q_min_pu": np.asarray(q_min, dtype=np.float32),
            "q_max_pu": np.asarray(q_max, dtype=np.float32),
        },
    )

    generator_state_edge = Edge.from_dict(
        address_dict={"bus": np.asarray(gen_bus, dtype=np.int32)},
        feature_dict={
            "p_pu": np.asarray(g_p, dtype=np.float32),
            "q_pu": np.asarray(g_q, dtype=np.float32),
            "vm_pu": np.asarray(g_vm, dtype=np.float32),
            "va_deg": np.asarray(g_va, dtype=np.float32),
        },
    )

    load_bus, l_p, l_q, l_vm, l_va = [], [], [], [], []
    if hasattr(net, "load") and len(net.load) > 0:
        for _, row in net.load.iterrows():
            b = int(row.bus)
            load_bus.append(bus_map[b])
            l_p.append(-float(row.p_mw) / base_mva)
            l_q.append(-float(row.q_mvar) / base_mva)
            l_vm.append(float(net.res_bus.loc[b, "vm_pu"]))
            l_va.append(float(net.res_bus.loc[b, "va_degree"]))

    load_edge = Edge.from_dict(
        address_dict={"bus": np.asarray(load_bus, dtype=np.int32)},
        feature_dict={"p_set_pu": np.asarray(l_p, dtype=np.float32), "q_set_pu": np.asarray(l_q, dtype=np.float32)},
    )

    load_state_edge = Edge.from_dict(
        address_dict={"bus": np.asarray(load_bus, dtype=np.int32)},
        feature_dict={
            "p_pu": np.asarray(l_p, dtype=np.float32),
            "q_pu": np.asarray(l_q, dtype=np.float32),
            "vm_pu": np.asarray(l_vm, dtype=np.float32),
            "va_deg": np.asarray(l_va, dtype=np.float32),
        },
    )

    p_from, q_from, p_to, q_to = [], [], [], []
    if hasattr(net, "res_line") and len(net.res_line) > 0:
        for _, row in net.res_line.iterrows():
            p_from.append(float(row.p_from_mw) / base_mva)
            q_from.append(float(row.q_from_mvar) / base_mva)
            p_to.append(float(row.p_to_mw) / base_mva)
            q_to.append(float(row.q_to_mvar) / base_mva)
    if hasattr(net, "res_trafo") and len(net.res_trafo) > 0:
        for _, row in net.res_trafo.iterrows():
            p_from.append(float(row.p_hv_mw) / base_mva)
            q_from.append(float(row.q_hv_mvar) / base_mva)
            p_to.append(float(row.p_lv_mw) / base_mva)
            q_to.append(float(row.q_lv_mvar) / base_mva)

    line_state_edge = Edge.from_dict(
        address_dict={"from_bus": np.asarray(line_from, dtype=np.int32), "to_bus": np.asarray(line_to, dtype=np.int32)},
        feature_dict={
            "p_from_pu": np.asarray(p_from, dtype=np.float32),
            "q_from_pu": np.asarray(q_from, dtype=np.float32),
            "p_to_pu": np.asarray(p_to, dtype=np.float32),
            "q_to_pu": np.asarray(q_to, dtype=np.float32),
        },
    )

    context_graph = Graph.from_dict(
        edge_dict={"lines": line_edge, "generators": generator_edge, "loads": load_edge},
        registry=registry,
    )
    oracle_graph = Graph.from_dict(
        edge_dict={"lines": line_state_edge, "generators": generator_state_edge, "loads": load_state_edge},
        registry=registry,
    )
    return problem_cls(context=context_graph, oracle=oracle_graph)

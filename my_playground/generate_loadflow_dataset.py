import argparse
import pickle
from pathlib import Path

import numpy as np
import pandapower as pp


def create_random_3bus_net(seed: int):
    rng = np.random.default_rng(seed)

    net = pp.create_empty_network(sn_mva=100.0)

    b0 = pp.create_bus(net, vn_kv=110.0, name="bus_slack")
    b1 = pp.create_bus(net, vn_kv=110.0, name="bus_load")
    b2 = pp.create_bus(net, vn_kv=110.0, name="bus_pv")

    pp.create_line_from_parameters(
        net,
        from_bus=b0,
        to_bus=b1,
        length_km=1.0,
        r_ohm_per_km=float(rng.uniform(0.06, 0.18)),
        x_ohm_per_km=float(rng.uniform(0.24, 0.62)),
        c_nf_per_km=float(rng.uniform(6.0, 18.0)),
        max_i_ka=float(rng.uniform(0.7, 1.4)),
        name="line_0_1",
    )
    pp.create_line_from_parameters(
        net,
        from_bus=b1,
        to_bus=b2,
        length_km=1.0,
        r_ohm_per_km=float(rng.uniform(0.06, 0.18)),
        x_ohm_per_km=float(rng.uniform(0.24, 0.62)),
        c_nf_per_km=float(rng.uniform(6.0, 18.0)),
        max_i_ka=float(rng.uniform(0.7, 1.4)),
        name="line_1_2",
    )

    pp.create_transformer_from_parameters(
        net,
        hv_bus=b0,
        lv_bus=b2,
        sn_mva=float(rng.uniform(80.0, 160.0)),
        vn_hv_kv=110.0,
        vn_lv_kv=110.0,
        vk_percent=float(rng.uniform(8.0, 14.0)),
        vkr_percent=float(rng.uniform(0.2, 1.0)),
        pfe_kw=0.0,
        i0_percent=0.0,
        shift_degree=float(rng.uniform(-4.0, 4.0)),
        tap_side="hv",
        tap_neutral=0,
        tap_min=-6,
        tap_max=6,
        tap_step_percent=1.25,
        tap_pos=int(rng.integers(-2, 3)),
        name="trafo_0_2",
    )

    p_load = float(rng.uniform(450.0, 950.0))
    q_load = float(rng.uniform(120.0, 360.0))
    p_pv = float(rng.uniform(0.35 * p_load, 0.85 * p_load))

    pp.create_ext_grid(net, bus=b0, vm_pu=1.0, va_degree=0.0, name="slack")
    pp.create_gen(
        net,
        bus=b2,
        p_mw=p_pv,
        vm_pu=float(rng.uniform(0.99, 1.04)),
        min_q_mvar=-400.0,
        max_q_mvar=400.0,
        name="pv_gen",
    )
    pp.create_load(net, bus=b1, p_mw=p_load, q_mvar=q_load, name="pq_load")

    pp.runpp(
        net,
        algorithm="nr",
        numba=True,
        calculate_voltage_angles=True,
        enforce_q_lims=True,
        init="flat",
        trafo_model="pi",
    )

    return net


def generate_dataset(output_path: Path, n_samples: int, seed: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    sample_seeds = rng.integers(low=0, high=np.iinfo(np.int32).max, size=n_samples, dtype=np.int32)
    nets = []

    for sample_seed in sample_seeds:
        net = create_random_3bus_net(int(sample_seed))
        nets.append(net)

    payload = {
        "format": "loadflow_3bus_dataset_pickle_v1",
        "n_samples": int(n_samples),
        "seed": int(seed),
        "nets": nets,
    }
    with output_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate random 3-bus pandapower loadflow dataset.")
    parser.add_argument("--output-path", type=Path, default=Path("my_playground/data/loadflow_3bus_dataset.pkl"))
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(output_path=args.output_path, n_samples=args.n_samples, seed=args.seed)
    print(f"Generated {args.n_samples} samples in {args.output_path}")

import argparse
import contextlib
import os
import io
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandapower as pp


def _create_connected_lines(
    net,
    *,
    rng: np.random.Generator,
    load_buses: list[int],
    generator_buses: list[int],
    n_extra_edges: int,
) -> None:
    n_buses = len(load_buses) + len(generator_buses)
    edge_set: set[tuple[int, int]] = set()
    load_ids = list(load_buses)

    if len(load_ids) == 0:
        raise ValueError("At least one load bus is required.")

    for idx in range(1, len(load_ids)):
        parent_idx = int(rng.integers(0, idx))
        from_bus = int(load_ids[parent_idx])
        to_bus = int(load_ids[idx])
        u, v = sorted((from_bus, to_bus))
        edge_set.add((u, v))
        pp.create_line_from_parameters(
            net,
            from_bus=from_bus,
            to_bus=to_bus,
            length_km=float(rng.uniform(0.5, 8.0)),
            r_ohm_per_km=float(rng.uniform(0.04, 0.22)),
            x_ohm_per_km=float(rng.uniform(0.16, 0.84)),
            c_nf_per_km=float(rng.uniform(3.0, 20.0)),
            max_i_ka=float(rng.uniform(0.4, 1.8)),
            name=f"line_{from_bus}_{to_bus}",
        )

    for generator_bus in generator_buses:
        neighbor_load = int(load_ids[int(rng.integers(0, len(load_ids)))])
        u, v = sorted((int(generator_bus), neighbor_load))
        edge_set.add((u, v))
        pp.create_line_from_parameters(
            net,
            from_bus=int(generator_bus),
            to_bus=neighbor_load,
            length_km=float(rng.uniform(0.5, 8.0)),
            r_ohm_per_km=float(rng.uniform(0.04, 0.22)),
            x_ohm_per_km=float(rng.uniform(0.16, 0.84)),
            c_nf_per_km=float(rng.uniform(3.0, 20.0)),
            max_i_ka=float(rng.uniform(0.4, 1.8)),
            name=f"line_{int(generator_bus)}_{neighbor_load}",
        )

    max_load_edges = (len(load_ids) * (len(load_ids) - 1)) // 2
    existing_load_edges = max(0, len(load_ids) - 1)
    extra_budget = min(max(0, n_extra_edges), max_load_edges - existing_load_edges)
    attempts = 0
    while extra_budget > 0 and attempts < 20 * max(1, n_buses):
        if len(load_ids) <= 1:
            break
        a_idx = int(rng.integers(0, len(load_ids)))
        b_idx = int(rng.integers(0, len(load_ids)))
        if a_idx == b_idx:
            attempts += 1
            continue
        a = int(load_ids[a_idx])
        b = int(load_ids[b_idx])
        u, v = sorted((a, b))
        if (u, v) in edge_set:
            attempts += 1
            continue
        edge_set.add((u, v))
        pp.create_line_from_parameters(
            net,
            from_bus=u,
            to_bus=v,
            length_km=float(rng.uniform(0.5, 8.0)),
            r_ohm_per_km=float(rng.uniform(0.04, 0.22)),
            x_ohm_per_km=float(rng.uniform(0.16, 0.84)),
            c_nf_per_km=float(rng.uniform(3.0, 20.0)),
            max_i_ka=float(rng.uniform(0.4, 1.8)),
            name=f"line_{u}_{v}",
        )
        extra_budget -= 1
        attempts += 1


def _add_generators_and_loads(
    net,
    *,
    rng: np.random.Generator,
    load_buses: list[int],
    generator_buses: list[int],
    slack_bus: int,
    required_load_buses: list[int],
    stress_scale: float,
) -> None:
    base_mva = float(getattr(net, "sn_mva", 100.0))

    pp.create_ext_grid(
        net,
        bus=slack_bus,
        vm_pu=float(rng.uniform(0.99, 1.01)),
        va_degree=0.0,
        name="slack",
    )

    if len(load_buses) == 0:
        return

    required_set = {int(bus) for bus in required_load_buses if int(bus) in set(load_buses)}
    optional_buses = [int(bus) for bus in load_buses if int(bus) not in required_set]

    if len(optional_buses) > 0:
        n_optional = int(np.round(rng.uniform(0.25, 0.75) * len(optional_buses)))
        n_optional = int(np.clip(n_optional, 0, len(optional_buses)))
        optional_selected = (
            rng.choice(np.asarray(optional_buses, dtype=np.int32), size=n_optional, replace=False).astype(np.int32).tolist()
        )
    else:
        optional_selected = []

    selected_load_buses = sorted(required_set.union({int(x) for x in optional_selected}))
    if len(selected_load_buses) == 0 and len(load_buses) > 0:
        selected_load_buses = [int(rng.choice(np.asarray(load_buses, dtype=np.int32)))]

    p_load_pu = rng.uniform(0.08 * stress_scale, 1.40 * stress_scale, size=len(selected_load_buses))
    power_factor = rng.uniform(0.92, 0.99, size=len(selected_load_buses))
    phi_rad = np.arccos(np.clip(power_factor, 1e-6, 1.0))
    q_over_p = np.tan(phi_rad)
    q_load_pu = p_load_pu * q_over_p
    for bus, p_pu, q_pu in zip(selected_load_buses, p_load_pu, q_load_pu):
        pp.create_load(
            net,
            bus=int(bus),
            p_mw=float(p_pu * base_mva),
            q_mvar=float(q_pu * base_mva),
            name=f"load_bus_{int(bus)}",
        )

    pv_buses = [int(bus) for bus in generator_buses if int(bus) != int(slack_bus)]
    if len(pv_buses) == 0:
        return

    total_load_pu = float(np.sum(p_load_pu))
    target_non_slack_gen_pu = float(rng.uniform(0.35, 0.85) * total_load_pu)
    gen_weights = rng.uniform(0.4, 1.6, size=len(pv_buses))
    gen_weights = gen_weights / float(np.sum(gen_weights))
    p_gen_pu = target_non_slack_gen_pu * gen_weights

    for bus, p_pu in zip(pv_buses, p_gen_pu):
        q_cap_pu = float(max(0.10, 0.8 * p_pu))
        pp.create_gen(
            net,
            bus=int(bus),
            p_mw=float(p_pu * base_mva),
            vm_pu=float(rng.uniform(0.985, 1.04)),
            min_q_mvar=float(-q_cap_pu * base_mva),
            max_q_mvar=float(q_cap_pu * base_mva),
            name=f"pv_bus_{int(bus)}",
        )


def _runpp_with_optional_quiet(net, *, quiet: bool, **kwargs) -> None:
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            pp.runpp(net, **kwargs)
        return
    pp.runpp(net, **kwargs)


def create_random_power_net(seed: int, n_buses: int, quiet: bool = False):
    if n_buses < 2 or n_buses >= 1000:
        raise ValueError("n_buses must satisfy 2 <= n_buses < 1000.")

    for attempt in range(20):
        attempt_seed = int((seed + 7919 * attempt) % np.iinfo(np.int32).max)
        rng = np.random.default_rng(attempt_seed)
        stress_scale = max(0.45, 1.0 - 0.04 * attempt)

        net = pp.create_empty_network(sn_mva=100.0)
        bus_ids = [pp.create_bus(net, vn_kv=110.0, name=f"bus_{i}") for i in range(n_buses)]

        slack_bus = int(bus_ids[0])
        candidate = np.asarray(bus_ids[1:], dtype=np.int32)
        n_non_slack = len(candidate)
        max_pv_by_ratio = int(np.floor(0.25 * n_non_slack))
        max_pv_by_load_reserve = max(0, n_non_slack - 3)
        max_pv = max(0, min(max_pv_by_ratio, max_pv_by_load_reserve))
        n_pv = int(rng.integers(0, max_pv + 1)) if max_pv > 0 else 0
        pv_buses = rng.choice(candidate, size=n_pv, replace=False).tolist() if n_pv > 0 else []
        generator_buses = [slack_bus] + [int(x) for x in pv_buses]
        load_buses = [int(bus) for bus in bus_ids if int(bus) not in set(generator_buses)]

        if len(load_buses) == 0:
            continue

        n_extra_edges = int(np.clip(np.round(rng.uniform(0.05, 0.25) * n_buses), 0, n_buses))
        _create_connected_lines(
            net,
            rng=rng,
            load_buses=load_buses,
            generator_buses=generator_buses,
            n_extra_edges=n_extra_edges,
        )

        degree_by_bus = {int(bus): 0 for bus in bus_ids}
        if hasattr(net, "line") and len(net.line) > 0:
            for _, row in net.line.iterrows():
                degree_by_bus[int(row.from_bus)] += 1
                degree_by_bus[int(row.to_bus)] += 1

        generator_set = {int(bus) for bus in generator_buses}
        required_load_buses = [
            int(bus)
            for bus in bus_ids
            if degree_by_bus[int(bus)] <= 1 and int(bus) not in generator_set
        ]

        _add_generators_and_loads(
            net,
            rng=rng,
            load_buses=load_buses,
            generator_buses=generator_buses,
            slack_bus=slack_bus,
            required_load_buses=required_load_buses,
            stress_scale=stress_scale,
        )

        run_configs = [
            {"algorithm": "nr", "init": "dc", "enforce_q_lims": True},
            {"algorithm": "iwamoto_nr", "init": "dc", "enforce_q_lims": True},
            {"algorithm": "nr", "init": "flat", "enforce_q_lims": False},
        ]

        converged = False
        for cfg in run_configs:
            try:
                _runpp_with_optional_quiet(
                    net,
                    quiet=quiet,
                    algorithm=cfg["algorithm"],
                    numba=True,
                    calculate_voltage_angles=True,
                    enforce_q_lims=cfg["enforce_q_lims"],
                    init=cfg["init"],
                    trafo_model="pi",
                    max_iteration=40,
                    tolerance_mva=1e-7,
                )
                converged = True
                break
            except Exception:
                continue

        if converged:
            return net

    raise RuntimeError(f"Failed to generate a convergent random network after retries (seed={seed}, n_buses={n_buses}).")


def _generate_one_sample(sample_seed: int, n_buses: int, quiet: bool, max_attempts: int = 25):
    rng = np.random.default_rng(sample_seed)
    for attempt_idx in range(max_attempts):
        candidate_seed = int(rng.integers(low=0, high=np.iinfo(np.int32).max, dtype=np.int32))
        try:
            net = create_random_power_net(seed=candidate_seed, n_buses=n_buses, quiet=quiet)
            return net, attempt_idx + 1
        except RuntimeError:
            continue
    raise RuntimeError(
        f"Could not generate sample from seed={sample_seed} after {max_attempts} retries for n_buses={n_buses}."
    )


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def generate_dataset(
    output_path: Path,
    n_samples: int,
    seed: int,
    n_buses: int,
    quiet: bool = False,
    progress: bool = False,
    progress_every: int = 1,
    jobs: int = 1,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if jobs < 1:
        jobs = max(1, (os.cpu_count() or 1) - 1)

    seed_rng = np.random.default_rng(seed)
    sample_seeds = [
        int(seed_rng.integers(low=0, high=np.iinfo(np.int32).max, dtype=np.int32))
        for _ in range(n_samples)
    ]

    nets = [None] * n_samples
    progress_step = max(1, progress_every)
    started_at = time.perf_counter()

    if jobs == 1:
        for sample_idx, sample_seed in enumerate(sample_seeds):
            net, attempts_used = _generate_one_sample(sample_seed=sample_seed, n_buses=n_buses, quiet=quiet)
            nets[sample_idx] = net
            if progress and (sample_idx + 1 == n_samples or (sample_idx + 1) % progress_step == 0):
                completed = sample_idx + 1
                elapsed_s = time.perf_counter() - started_at
                rate = completed / max(elapsed_s, 1e-9)
                eta_s = (n_samples - completed) / max(rate, 1e-9)
                print(
                    f"Progress: generated sample {completed}/{n_samples} "
                    f"(last sample attempts: {attempts_used}/25, "
                    f"elapsed: {_format_duration(elapsed_s)}, eta: {_format_duration(eta_s)})",
                    flush=True,
                )
    else:
        completed = 0
        futures = {}
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            for sample_idx, sample_seed in enumerate(sample_seeds):
                future = executor.submit(_generate_one_sample, sample_seed, n_buses, quiet)
                futures[future] = sample_idx

            for future in as_completed(futures):
                sample_idx = futures[future]
                net, attempts_used = future.result()
                nets[sample_idx] = net
                completed += 1
                if progress and (completed == n_samples or completed % progress_step == 0):
                    elapsed_s = time.perf_counter() - started_at
                    rate = completed / max(elapsed_s, 1e-9)
                    eta_s = (n_samples - completed) / max(rate, 1e-9)
                    print(
                        f"Progress: generated sample {completed}/{n_samples} "
                        f"(last sample attempts: {attempts_used}/25, "
                        f"elapsed: {_format_duration(elapsed_s)}, eta: {_format_duration(eta_s)})",
                        flush=True,
                    )

    payload = {
        "format": "loadflow_random_nbus_dataset_pickle_v1",
        "n_samples": int(n_samples),
        "seed": int(seed),
        "n_buses": int(n_buses),
        "nets": nets,
    }
    with output_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def resolve_output_path(output_path: Path | None, *, n_samples: int, n_buses: int) -> Path:
    auto_name = f"loadflow_{n_buses}bus_{n_samples}.pkl"
    default_dir = Path("my_playground/data")

    if output_path is None:
        return default_dir / auto_name

    if output_path.suffix.lower() == ".pkl":
        return output_path

    return output_path / auto_name


def parse_args():
    parser = argparse.ArgumentParser(description="Generate random pandapower loadflow dataset for arbitrary bus count.")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-buses", type=int, default=32)
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose solver output during data generation.")
    parser.add_argument("--progress", action="store_true", help="Print periodic generation progress.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print progress every N samples when --progress is set.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker processes (1 = sequential, <1 = auto cores-1).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resolved_output_path = resolve_output_path(args.output_path, n_samples=args.n_samples, n_buses=args.n_buses)
    generate_dataset(
        output_path=resolved_output_path,
        n_samples=args.n_samples,
        seed=args.seed,
        n_buses=args.n_buses,
        quiet=args.quiet,
        progress=args.progress,
        progress_every=args.progress_every,
        jobs=args.jobs,
    )
    print(f"Generated {args.n_samples} samples (n_buses={args.n_buses}) in {resolved_output_path}")

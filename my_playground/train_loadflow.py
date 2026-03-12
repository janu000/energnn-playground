from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


class _EpochAverageLossTracker:
    def __init__(self, base_tracker, *, steps_per_epoch: int):
        self._base_tracker = base_tracker
        self._steps_per_epoch = max(1, int(steps_per_epoch))
        self._epoch_train_losses: list[float] = []

    def init_run(self, *, name, tags, cfg):
        return self._base_tracker.init_run(name=name, tags=tags, cfg=cfg)

    def stop_run(self):
        return self._base_tracker.stop_run()

    def get_amortizer_path(self, *, tag):
        return self._base_tracker.get_amortizer_path(tag=tag)

    def run_track_dataset(self, *, infos, target_path):
        return self._base_tracker.run_track_dataset(infos=infos, target_path=target_path)

    def run_track_amortizer(self, *, id, target_path):
        return self._base_tracker.run_track_amortizer(id=id, target_path=target_path)

    @staticmethod
    def _extract_train_loss(train_infos: dict) -> float | None:
        candidates = (
            "3_gradient/mse",
            "loss",
            "mse",
            "rmse",
            "metrics",
        )
        for key in candidates:
            if key in train_infos:
                try:
                    value = float(np.asarray(train_infos[key]).reshape(-1)[0])
                except Exception:
                    return None
                if np.isfinite(value):
                    return value
        return None

    def run_append(self, *, infos, step):
        train_infos = infos.get("train") if isinstance(infos, dict) else None
        if isinstance(train_infos, dict):
            train_loss = self._extract_train_loss(train_infos)
            if train_loss is not None:
                self._epoch_train_losses.append(train_loss)

        if (int(step) + 1) % self._steps_per_epoch == 0:
            epoch_idx = (int(step) + 1) // self._steps_per_epoch
            if self._epoch_train_losses:
                avg_loss = float(np.mean(self._epoch_train_losses))
                print(f"Epoch {epoch_idx}: avg_train_loss={avg_loss:.6e}")
            self._epoch_train_losses.clear()

        return self._base_tracker.run_append(infos=infos, step=step)

def _parse_hidden_sizes(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _split_train_val(dataset: list, val_fraction: float, seed: int) -> tuple[list, list]:
    if not dataset:
        raise ValueError("Dataset is empty.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1).")

    idx = np.arange(len(dataset))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_val = max(1, int(round(len(dataset) * val_fraction)))
    n_val = min(n_val, len(dataset) - 1)

    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_set = [dataset[int(i)] for i in train_idx]
    val_set = [dataset[int(i)] for i in val_idx]
    return train_set, val_set


def _build_tracker(args):
    if args.tracker_backend == "none" or args.tracker_backend == "dummy":
        from energnn.tracker.dummy import DummyTracker

        return DummyTracker()
    if args.tracker_backend == "neptune":
        if not args.project_name:
            raise ValueError("--project-name is required for tracker backend 'neptune'.")
        from energnn.tracker.neptune import NeptuneTracker

        return NeptuneTracker(project_name=args.project_name)
    if args.tracker_backend == "neptune_scale":
        if not args.project_name:
            raise ValueError("--project-name is required for tracker backend 'neptune_scale'.")
        from energnn.tracker.neptune_scale import NeptuneScaleTracker

        return NeptuneScaleTracker(project_name=args.project_name)
    raise ValueError(f"Unsupported tracker backend: {args.tracker_backend}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a modular load-flow model on a pickled dataset.")
    parser.add_argument("--dataset-path", type=Path, default=Path("my_playground/data/loadflow_3bus_pipeline_test.pkl"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--cpu-cores",
        type=int,
        default=0,
        help="Number of CPU cores/threads for JAX CPU execution (<=0 uses all available cores).",
    )

    parser.add_argument("--n-breakpoints", type=int, default=50)
    parser.add_argument("--latent-dimension", type=int, default=16)
    parser.add_argument("--hidden-sizes", type=str, default="32")
    parser.add_argument("--n-steps", type=int, default=20)

    parser.add_argument("--checkpoint-dir", type=Path, default=Path("my_playground/checkpoints/loadflow"))
    parser.add_argument("--tracker-backend", choices=["none", "dummy", "neptune", "neptune_scale"], default="dummy")
    parser.add_argument("--project-name", type=str, default="")
    parser.add_argument("--run-name", type=str, default="loadflow-custom-simplegnn")
    parser.add_argument("--tags", type=str, default="loadflow,playground")

    parser.add_argument("--log-period", type=int, default=1)
    parser.add_argument(
        "--eval-period",
        type=int,
        default=10,
        help="Evaluation period in epochs (set <=0 to disable periodic evaluation).",
    )
    parser.add_argument(
        "--eval-after-epoch",
        action="store_true",
        help="If set, run evaluation at the end of every epoch.",
    )
    return parser.parse_args()


def _configure_cpu_threads(requested_cpu_cores: int) -> int:
    available_cores = os.cpu_count() or 1
    if requested_cpu_cores <= 0:
        cpu_cores = available_cores
    else:
        cpu_cores = min(requested_cpu_cores, available_cores)

    existing_xla_flags = os.environ.get("XLA_FLAGS", "").strip()
    kept_flags = []
    if existing_xla_flags:
        for token in existing_xla_flags.split():
            if token.startswith("--xla_cpu_multi_thread_eigen"):
                continue
            if token.startswith("intra_op_parallelism_threads="):
                continue
            kept_flags.append(token)

    kept_flags.append("--xla_cpu_multi_thread_eigen=true")
    kept_flags.append(f"intra_op_parallelism_threads={cpu_cores}")
    os.environ["XLA_FLAGS"] = " ".join(kept_flags).strip()

    os.environ["OMP_NUM_THREADS"] = str(cpu_cores)
    os.environ["MKL_NUM_THREADS"] = str(cpu_cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_cores)

    return cpu_cores


def main() -> None:
    args = parse_args()
    configured_cpu_cores = _configure_cpu_threads(args.cpu_cores)

    import optax
    import orbax.checkpoint as ocp
    from omegaconf import OmegaConf

    from energnn.trainer import SimpleTrainer
    from loadflow_model import LoadFlowModelConfig, build_loadflow_model
    from loadflow_problem import LoadFlowDataLoader, LoadFlowProblem

    dataset = LoadFlowProblem.load_dataset(args.dataset_path)
    train_dataset, val_dataset = _split_train_val(dataset=dataset, val_fraction=args.val_fraction, seed=args.seed)

    train_loader = LoadFlowDataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_loader = LoadFlowDataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, seed=args.seed)

    model_cfg = LoadFlowModelConfig(
        n_breakpoints=args.n_breakpoints,
        latent_dimension=args.latent_dimension,
        hidden_sizes=_parse_hidden_sizes(args.hidden_sizes),
        n_steps=args.n_steps,
        seed=args.seed,
    )

    model = build_loadflow_model(
        in_structure=train_loader.context_structure,
        out_structure=train_loader.decision_structure,
        cfg=model_cfg,
    )

    trainer = SimpleTrainer(
        model=model,
        gradient_transformation=optax.adam(args.learning_rate),
    )

    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = ocp.CheckpointManager(directory=checkpoint_dir)

    tracker = _build_tracker(args)
    tracker = _EpochAverageLossTracker(tracker, steps_per_epoch=len(train_loader))
    cfg_dict = vars(args).copy()
    cfg_dict["effective_cpu_cores"] = configured_cpu_cores
    cfg = OmegaConf.create(cfg_dict)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    tracker.init_run(name=args.run_name, tags=tags, cfg=cfg)

    eval_period_steps = None
    if args.eval_period > 0:
        eval_period_steps = args.eval_period * len(train_loader)

    try:
        best_metrics = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_manager=checkpoint_manager,
            tracker=tracker,
            n_epochs=args.n_epochs,
            log_period=args.log_period,
            eval_period=eval_period_steps,
            eval_before_training=True,
            eval_after_epoch=args.eval_after_epoch,
            progress_bar=True,
        )
    finally:
        tracker.stop_run()

    print(f"Training finished. Best validation metric: {best_metrics:.6e}")


if __name__ == "__main__":
    main()

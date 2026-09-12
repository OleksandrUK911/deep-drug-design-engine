"""Experiment tracking wrapper around a local, self-hosted MLflow store
(see TODO/ml/TODO_experiment_tracking.md).

Uses a file-based backend (`./mlruns/`, gitignored) so tracking works with
zero external account/service — every training run in this project should
go through `tracked_run()` from its very first invocation, not be bolted on
later. Swapping to a remote MLflow server or W&B later is a one-line change
to `_TRACKING_URI`; the rest of the training code does not need to change.
"""
from __future__ import annotations

import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import mlflow

REPO_ROOT = Path(__file__).resolve().parent.parent
# MLflow 3.x put the plain filesystem store into maintenance mode; a local
# sqlite file is the supported zero-account, self-hosted equivalent.
_TRACKING_URI = f"sqlite:///{(REPO_ROOT / 'mlruns.db').as_posix()}"

mlflow.set_tracking_uri(_TRACKING_URI)


def _git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _file_sha256_prefix(path: Path, length: int = 12) -> str:
    import hashlib

    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


@contextmanager
def tracked_run(
    experiment_name: str,
    run_name: str,
    params: dict,
    dataset_path: Optional[Path] = None,
) -> Iterator["mlflow.ActiveRun"]:
    """Start an MLflow run under `experiment_name`, automatically logging
    hyperparameters, the current git commit hash, and (if given) the
    dataset file's checksum — so every run is traceable back to exactly
    the code and data version that produced it.
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.set_tag("git_commit", _git_commit_hash())
        if dataset_path is not None:
            mlflow.set_tag("dataset_file", str(dataset_path))
            mlflow.set_tag("dataset_sha256_12", _file_sha256_prefix(Path(dataset_path)))
        yield run


def log_metrics(metrics: dict, step: Optional[int] = None) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: Path) -> None:
    mlflow.log_artifact(str(path))


def mark_best(run_id: str, model_type: str) -> None:
    """Tag a run as the current-best checkpoint for a given model type
    (e.g. "gnn_property"), so ml/TODO_model_registry.md has a single
    source of truth for "which run do I promote to the registry."
    """
    client = mlflow.tracking.MlflowClient()
    client.set_tag(run_id, "best_for", model_type)

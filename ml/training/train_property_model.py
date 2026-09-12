"""Train a baseline GNN (GCN or GIN) on a MoleculeNet-style property CSV,
with scaffold split, early stopping, and MLflow experiment tracking.
See TODO/ml/TODO_gnn_property_model.md and TODO/ml/TODO_experiment_tracking.md.

The core logic lives in `run_training()` so it can be reused both by this
CLI and by ml/training/low_data_experiment.py (SSL-pretrained vs
from-scratch comparison at varying training-set fractions).

Usage:
    python -m ml.training.train_property_model --dataset data/raw/BBBP.csv \
        --smiles-col smiles --label-col p_np --task classification --architecture gcn

    python -m ml.training.train_property_model --dataset data/raw/delaney-processed.csv \
        --smiles-col smiles --label-col "measured log solubility in mols per litre" \
        --task regression --architecture gin

    python -m ml.training.train_property_model --dataset data/raw/BBBP.csv \
        --smiles-col smiles --label-col p_np --task classification --architecture gin \
        --pretrained-encoder mlruns_checkpoints/ssl_encoder.pt
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from torch_geometric.loader import DataLoader

from ml.data.dataset import load_smiles_and_labels
from ml.data.featurizer import NUM_ATOM_FEATURES, smiles_to_graph
from ml.data.splits import scaffold_split
from ml.models.gnn import GINPropertyModel, build_model
from ml.tracking import log_artifact, log_metrics, tracked_run


def build_graphs(smiles: list[str], labels: np.ndarray) -> list:
    graphs = []
    for s, y in zip(smiles, labels):
        g = smiles_to_graph(s, y=float(y))
        if g is not None:
            graphs.append(g)
    return graphs


def evaluate(model, loader, task: str, device: str) -> dict:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
            preds.append(out.cpu().numpy())
            targets.append(batch.y.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    if task == "classification":
        probs = 1 / (1 + np.exp(-preds))
        return {"roc_auc": float(roc_auc_score(targets, probs))}
    return {
        "rmse": float(mean_squared_error(targets, preds) ** 0.5),
        "r2": float(r2_score(targets, preds)),
    }


def run_training(
    smiles: list[str],
    labels: np.ndarray,
    task: str,
    architecture: str = "gin",
    hidden_dim: int = 64,
    num_layers: int = 3,
    lr: float = 1e-3,
    epochs: int = 100,
    patience: int = 10,
    batch_size: int = 64,
    seed: int = 0,
    dropout: float = 0.0,
    pretrained_encoder_path: Optional[Path] = None,
    train_frac: float = 1.0,
    device: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Split -> (optionally subsample train) -> build model (optionally
    loading a pretrained GIN encoder) -> train with early stopping ->
    return test metrics plus the sizes actually used, so low-data
    experiments can report both."""
    torch.manual_seed(seed)
    random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_idx, valid_idx, test_idx = scaffold_split(smiles, 0.8, 0.1, 0.1, seed=seed)

    if train_frac < 1.0:
        rng = random.Random(seed)
        n_keep = max(1, int(len(train_idx) * train_frac))
        train_idx = rng.sample(train_idx, n_keep)

    def subset(indices):
        return build_graphs([smiles[i] for i in indices], labels[indices])

    train_graphs, valid_graphs, test_graphs = subset(train_idx), subset(valid_idx), subset(test_idx)

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_graphs, batch_size=batch_size)
    test_loader = DataLoader(test_graphs, batch_size=batch_size)

    model = build_model(architecture, in_dim=NUM_ATOM_FEATURES, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)

    if pretrained_encoder_path is not None:
        if not isinstance(model, GINPropertyModel):
            raise ValueError("--pretrained-encoder currently only supports --architecture gin")
        state_dict = torch.load(pretrained_encoder_path, map_location=device)
        model.load_pretrained_encoder(state_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss() if task == "classification" else torch.nn.MSELoss()

    best_valid_metric = None
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        train_loss = total_loss / max(len(train_graphs), 1)
        valid_metrics = evaluate(model, valid_loader, task, device)
        key_metric = valid_metrics["roc_auc"] if task == "classification" else -valid_metrics["rmse"]

        log_metrics({"train_loss": train_loss, **{f"valid_{k}": v for k, v in valid_metrics.items()}}, step=epoch)
        if verbose:
            print(f"epoch {epoch:3d} | train_loss={train_loss:.4f} | valid={valid_metrics}")

        if best_valid_metric is None or key_metric > best_valid_metric:
            best_valid_metric = key_metric
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, task, device)
    if verbose:
        print(f"Test metrics: {test_metrics}")
    log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    return {
        "test_metrics": test_metrics,
        "n_train": len(train_graphs),
        "n_valid": len(valid_graphs),
        "n_test": len(test_graphs),
        "model": model,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--architecture", choices=["gcn", "gin"], default="gcn")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1, help="also enables MC Dropout at inference time")
    parser.add_argument("--pretrained-encoder", type=Path, default=None)
    parser.add_argument("--train-frac", type=float, default=1.0)
    args = parser.parse_args()

    smiles, labels_df = load_smiles_and_labels(args.dataset, args.smiles_col, [args.label_col])
    labels = labels_df[args.label_col].to_numpy(dtype=float)

    print(f"Split sizes will be computed inside run_training (train_frac={args.train_frac})")

    params = vars(args) | {"num_atom_features": NUM_ATOM_FEATURES}
    with tracked_run(f"gnn_property_{args.task}", f"{args.architecture}_{args.dataset.stem}", params, args.dataset):
        result = run_training(
            smiles, labels, args.task, args.architecture, args.hidden_dim, args.num_layers,
            args.lr, args.epochs, args.patience, args.batch_size, args.seed, args.dropout,
            args.pretrained_encoder, args.train_frac,
        )

        checkpoint_dir = Path("mlruns_checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        ckpt_path = checkpoint_dir / f"{args.architecture}_{args.dataset.stem}.pt"
        torch.save(result["model"].state_dict(), ckpt_path)
        log_artifact(ckpt_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

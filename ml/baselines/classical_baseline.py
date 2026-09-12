"""Classical RDKit-descriptor + XGBoost baseline, evaluated on the exact
same scaffold split as the GNN models — the comparison required by
TODO/ml/TODO_gnn_property_model.md ("Порівняти GCN/GIN/MPNN проти
класичного baseline"). Because `scaffold_split` is a deterministic
function of the SMILES list and seed, running it here reproduces the
identical train/valid/test partition used by
ml/training/train_property_model.py, so the comparison is apples-to-apples.

Usage:
    python -m ml.baselines.classical_baseline --dataset data/raw/BBBP.csv \
        --smiles-col smiles --label-col p_np --task classification
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

from ml.data.dataset import load_smiles_and_labels
from ml.data.splits import scaffold_split
from ml.tracking import log_metrics, tracked_run

# A compact, standard subset of RDKit's 2D descriptors — enough to be a
# credible classical baseline without hand-tuning a huge feature set.
DESCRIPTOR_FNS = [
    Descriptors.MolWt,
    Descriptors.MolLogP,
    Descriptors.NumHAcceptors,
    Descriptors.NumHDonors,
    Descriptors.NumRotatableBonds,
    Descriptors.TPSA,
    Descriptors.RingCount,
    Descriptors.NumAromaticRings,
    Descriptors.FractionCSP3,
    Descriptors.HeavyAtomCount,
]


def descriptors(smiles: str) -> list[float]:
    mol = Chem.MolFromSmiles(smiles)
    return [fn(mol) for fn in DESCRIPTOR_FNS]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    smiles, labels_df = load_smiles_and_labels(args.dataset, args.smiles_col, [args.label_col])
    labels = labels_df[args.label_col].to_numpy(dtype=float)
    train_idx, valid_idx, test_idx = scaffold_split(smiles, 0.8, 0.1, 0.1, seed=args.seed)

    X = np.array([descriptors(s) for s in smiles])
    X_train, X_valid, X_test = X[train_idx], X[valid_idx], X[test_idx]
    y_train, y_valid, y_test = labels[train_idx], labels[valid_idx], labels[test_idx]

    params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05, "seed": args.seed}
    with tracked_run(f"classical_baseline_{args.task}", f"xgboost_{args.dataset.stem}", params, args.dataset):
        if args.task == "classification":
            model = XGBClassifier(**params, eval_metric="auc")
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            probs = model.predict_proba(X_test)[:, 1]
            metrics = {"test_roc_auc": float(roc_auc_score(y_test, probs))}
        else:
            model = XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            preds = model.predict(X_test)
            metrics = {
                "test_rmse": float(mean_squared_error(y_test, preds) ** 0.5),
                "test_r2": float(r2_score(y_test, preds)),
            }

        print(f"Classical baseline (RDKit descriptors + XGBoost) test metrics: {metrics}")
        log_metrics(metrics)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Low-data experiment (see TODO/ml/TODO_ssl_pretraining.md): train GIN
from scratch vs. fine-tuned from the SSL-pretrained encoder
(ml/training/pretrain_ssl.py), at several training-set fractions, on the
identical scaffold split — the most convincing evidence for whether SSL
pretraining actually helps, since the gap should be largest exactly when
labeled data is scarce.

Usage:
    python -m ml.training.low_data_experiment --dataset data/raw/BBBP.csv \
        --smiles-col smiles --label-col p_np --task classification \
        --pretrained-encoder mlruns_checkpoints/ssl_encoder.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.data.dataset import load_smiles_and_labels
from ml.tracking import tracked_run
from ml.training.train_property_model import run_training

FRACTIONS = [0.1, 0.25, 0.5, 1.0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--pretrained-encoder", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    smiles, labels_df = load_smiles_and_labels(args.dataset, args.smiles_col, [args.label_col])
    labels = labels_df[args.label_col].to_numpy(dtype=float)

    results = []
    for frac in FRACTIONS:
        for condition, encoder_path in [("scratch", None), ("ssl_pretrained", args.pretrained_encoder)]:
            run_name = f"lowdata_{condition}_frac{frac}"
            with tracked_run("low_data_ssl_comparison", run_name, {"train_frac": frac, "condition": condition}, args.dataset):
                result = run_training(
                    smiles, labels, args.task, architecture="gin",
                    epochs=args.epochs, patience=args.patience, seed=args.seed,
                    dropout=0.1, pretrained_encoder_path=encoder_path, train_frac=frac,
                    verbose=False,
                )
            metric_name = "roc_auc" if args.task == "classification" else "rmse"
            value = result["test_metrics"][metric_name]
            results.append((frac, condition, value, result["n_train"]))
            print(f"frac={frac:>4} condition={condition:15s} n_train={result['n_train']:4d} test_{metric_name}={value:.4f}")

    print("\n| train_frac | n_train | scratch | ssl_pretrained | delta |")
    print("|---|---|---|---|---|")
    by_frac = {}
    for frac, condition, value, n_train in results:
        by_frac.setdefault(frac, {})[condition] = (value, n_train)
    for frac in FRACTIONS:
        scratch_val, n_train = by_frac[frac]["scratch"]
        ssl_val, _ = by_frac[frac]["ssl_pretrained"]
        delta = ssl_val - scratch_val
        print(f"| {frac} | {n_train} | {scratch_val:.4f} | {ssl_val:.4f} | {delta:+.4f} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

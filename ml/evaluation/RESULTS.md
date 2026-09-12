# Baseline results — BBBP (blood-brain barrier penetration)

Same scaffold split for every model (`scaffold_split(seed=0)` on the
deduplicated BBBP SMILES: 1580 train / 197 valid / 198 test, zero scaffold
overlap between sides — see `TODO/data/TODO_quality_validation.md`).
Tracked in MLflow (`mlruns.db`, local/gitignored); reproduce with the
commands below.

| Model | Features | Test ROC-AUC |
|---|---|---|
| XGBoost | 10 RDKit 2D descriptors (MolWt, LogP, TPSA, ring counts, ...) | 0.816 |
| GCN | learned graph features (3-layer GCNConv + mean pool) | 0.832 |
| GIN | learned graph features (3-layer GINConv + mean pool) | **0.912** |

## Reading

- Both GNNs beat the classical descriptor baseline, but the gap between
  GCN and GIN (0.832 -> 0.912) is much larger than the gap between XGBoost
  and GCN (0.816 -> 0.832). This matches the theoretical motivation for
  GIN (Xu et al., 2019): it is provably more expressive at distinguishing
  non-isomorphic graph structures than GCN's mean/sum aggregation.
- This is a single scaffold-split run (seed=0), not an average over
  multiple seeds — `ml/TODO_gnn_property_model.md`'s hyperparameter-search
  task and `ml/TODO_ablation_active_learning.md` are where this gets
  turned into a statistically defensible comparison (multiple seeds,
  confidence intervals) rather than a single anecdotal number.

## ESOL (aqueous solubility, regression)

Single run, GIN, same scaffold-split methodology (seed=0):

| Model | Test RMSE | Test R² |
|---|---|---|
| GIN | 1.29 | 0.59 |

Confirms the regression path of the training loop and `evaluate()` end to
end; classical-baseline and GCN regression runs, and a multi-seed average,
are not done yet.

## Reproduce

```bash
python -m ml.baselines.classical_baseline --dataset data/raw/BBBP.csv \
    --smiles-col smiles --label-col p_np --task classification

python -m ml.training.train_property_model --dataset data/raw/BBBP.csv \
    --smiles-col smiles --label-col p_np --task classification --architecture gcn

python -m ml.training.train_property_model --dataset data/raw/BBBP.csv \
    --smiles-col smiles --label-col p_np --task classification --architecture gin
```

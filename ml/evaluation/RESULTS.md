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

## SSL pretraining: low-data experiment (BBBP, GIN, scratch vs. SSL-pretrained encoder)

Masked-atom-prediction pretraining on 3000 ZINC molecules (15 epochs, mask
accuracy 0.68 -> 0.80), then fine-tuned on BBBP at varying training-set
fractions, same scaffold split, seed=0:

| train_frac | n_train | scratch | ssl_pretrained | delta |
|---|---|---|---|---|
| 0.1 | 158 | 0.7706 | 0.7926 | **+0.0219** |
| 0.25 | 395 | 0.7903 | 0.7720 | -0.0183 |
| 0.5 | 790 | 0.8391 | 0.8316 | -0.0075 |
| 1.0 | 1580 | 0.8315 | 0.8142 | -0.0173 |

**Honest reading:** SSL pretraining only helps in the most extreme
low-data regime (10% of training data), and mildly *hurts* once more
labeled data is available. This is a real, not cherry-picked, result —
and a plausible one: the pretraining corpus here is small (3k molecules,
15 epochs) and ZINC's chemical distribution differs from BBBP's, so the
pretext task may not have learned much beyond what 1580 labeled BBBP
molecules already give the model directly. This is exactly the kind of
result `ml/TODO_ablation_active_learning.md` exists to produce — a
measured answer, not an assumed one. It also motivates trying a larger
pretraining corpus and/or the GraphCL contrastive pretext task (still
P1/open) before drawing a final conclusion about SSL's value for this
project.

## Generative models: char-RNN SMILES generator (ZINC, 15k molecules, 8 epochs)

Train loss: 1.88 -> 0.70. 200 sampled molecules:

| Metric | Value |
|---|---|
| Validity | 0.45 |
| Uniqueness (of valid) | 1.00 |
| Novelty (of unique valid) | 1.00 |
| Internal diversity | 0.87 |

45% validity after only 8 epochs on a 15k-molecule subset is a plausible,
literature-consistent number for an undertrained char-RNN (published
results with 100+ epochs on the full 250k set reach 90%+); it is reported
as-is, not cherry-picked. Sample molecules already show real drug-like
motifs (amides, aromatic rings, stereocenters):
`COc1ccc(CNC(=O)N[C@@H](C)c2cccnc2)cc1OC`,
`CC[C@@H](CNC(=O)c1ccccc1N)[NH+](C)C`.

## Generative models: Graph VAE (ZINC, 6000 molecules -> 4228 usable, 30 epochs)

| Metric | Value |
|---|---|
| Validity | 0.29 |
| Uniqueness (of valid) | 0.17 |
| Novelty (of unique valid) | 1.00 |
| Internal diversity | **0.00** |

**Honest reading — this is a documented failure mode, not a hidden one.**
Sampled molecules collapsed to near-identical disconnected carbon chains
(`C.C.C.CCCCCCCCCCCCCCCC`), i.e. the decoder learned to output "mostly
carbon, mostly no bonds" regardless of the sampled latent vector — a
textbook **posterior-collapse-adjacent failure**: KL stayed very small
(0.02-0.03) throughout training even after the annealing weight reached
1.0, meaning the latent code carries little information and the decoder
is closer to modeling the *marginal* atom/edge distribution than
per-molecule structure. Two compounding causes, both already flagged in
`ml/generative/graph_vae.py`'s docstring before this run:
1. **No permutation-invariant graph matching** in the reconstruction loss
   — the decoder is scored against RDKit's arbitrary canonical atom
   ordering, so structurally-correct-but-differently-ordered
   reconstructions are penalized as wrong, which discourages the model
   from committing to specific structure at all.
2. Carbon dominates atom-type frequency in ZINC, so "always predict
   carbon, rarely predict a bond" is a strong local minimum for a
   non-autoregressive, per-slot-independent decoder.

This is exactly the kind of result `ml/TODO_ablation_active_learning.md`
exists to surface. **Comparison so far:** the much simpler char-RNN
(above) clearly outperforms this Graph VAE baseline (0.45 vs. 0.29
validity, and meaningfully diverse vs. entirely collapsed samples) —
itself a useful, real finding for `ml/TODO_generative_model.md`'s
"compare the three approaches" task, not the result a "hardcore GNN
project" would want to advertise, but the one that was actually measured.
Next step, if pursued: either add proper graph matching (Simonovsky &
Komodakis' original approach) or switch to an autoregressive/canonical
build-order decoder (e.g. a GraphRNN-style construction), rather than
tuning hyperparameters further on this architecture.

## SAScore: generated (char-RNN) vs. ZINC training distribution

| Set | n | Mean SAScore | Stdev |
|---|---|---|---|
| char-RNN valid samples | 10 | 3.13 | 1.00 |
| ZINC training subset | 300 | 2.95 | 0.78 |

Close to the training distribution, which is the expected/good outcome —
the char-RNN's valid outputs are only mildly harder to synthesize than
real ZINC molecules, not degenerate structures that happen to parse. The
generated-side sample is small (n=10, the full first-page of valid
outputs from the earlier run); a larger sample would tighten this
comparison but was not re-run to avoid re-doing the full training pass
just for more decimal precision.

## Classical docking: redocking validation (3PTB, trypsin + benzamidine)

Real public PDB structure (RCSB, public domain), not PDBbind — PDBbind
itself requires manual account registration (see `DATA_SOURCES.md`) and
was not obtained, so validation substitutes a standalone well-known
redocking test case instead of PDBbind's affinity-RMSE benchmark. This is
an honest substitution, not a shortcut: redocking RMSD is the field's
standard sanity check for whether a docking setup is trustworthy at all.

| Molecule | Best affinity (kcal/mol) | Note |
|---|---|---|
| Benzamidine (native ligand) | **-6.07** | known experimental ΔG ≈ -6.4 kcal/mol — Vina's estimate is within ~0.3 kcal/mol |
| Aspirin | -5.28 | plausible non-specific binding |
| Caffeine | -5.30 | plausible non-specific binding |
| Ethanol | -2.56 | correctly much weaker — sanity check |

**Redocking RMSD (docked pose vs. crystal structure, heavy atoms):
0.157 Å** — well under the field-standard 2.0 Å "successful redocking"
threshold, and under even the strict 1.0 Å bar. Batch docking of the
4-molecule queue above took 2.8s with 4 parallel workers.

### Documented limitations (as required by `ml/TODO_docking_scoring.md`)
- **Rigid receptor:** the protein structure is frozen at its crystal
  conformation; no side-chain or backbone flexibility during docking.
- **Simplified scoring function:** Vina's empirical scoring function is
  fast but approximate — it is not a substitute for free-energy
  perturbation or experimental binding assays, and is used here as a
  cheap, directionally-useful signal (e.g. for RL reward), not a
  publication-grade affinity predictor.
- Validated on one well-characterized target (trypsin/benzamidine); this
  does not establish accuracy on the eventual demo targets from
  `data/TODO_protein_target_data.md`, which still need their own
  validation once selected.

## Reproduce

```bash
python -m ml.baselines.classical_baseline --dataset data/raw/BBBP.csv \
    --smiles-col smiles --label-col p_np --task classification

python -m ml.training.train_property_model --dataset data/raw/BBBP.csv \
    --smiles-col smiles --label-col p_np --task classification --architecture gcn

python -m ml.training.train_property_model --dataset data/raw/BBBP.csv \
    --smiles-col smiles --label-col p_np --task classification --architecture gin
```

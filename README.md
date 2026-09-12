# Deep Generative Drug Design Engine

A deep-learning-first drug discovery system: Graph Neural Networks predict
molecular properties, a generative model designs novel candidate molecules
from scratch, reinforcement learning steers generation toward drug-like,
high-scoring compounds, and a docking oracle (classical + generative) scores
candidates against a target binding pocket.

This is the most research-heavy project in the series — the point is to go
deep on modern molecular deep learning (GNNs, self-supervised pretraining,
generative models, RL, 3D/equivariant networks) rather than to ship the
broadest product surface.

## Status
🔴 In planning — implementation not started yet.

## Stack
- **ML:** Python, PyTorch, PyTorch Geometric (GNN: GCN/GIN/MPNN, EGNN),
  RDKit, self-supervised pretraining (GraphCL/masked-attribute), Graph
  VAE / discrete diffusion for molecule generation, REINFORCE/PPO for
  goal-directed optimization, AutoDock Vina for classical docking,
  diffusion-based generative docking (DiffDock-style) for pose prediction
- **Backend:** FastAPI, PostgreSQL, async job queue (Celery/RQ) for
  GPU-bound training/generation/docking jobs
- **Frontend:** React, 2D structure + 3D binding-pose viewer, generation
  and training-run dashboards
- **Infra:** Docker (GPU-enabled), GitHub Actions, model registry

## How it works
```
Target property/pocket spec
  → GNN property predictor (pretrained + fine-tuned, with uncertainty)
  → generative model proposes candidate molecules
  → RL loop optimizes candidates against predictor + docking reward
  → classical + generative docking score binding pose
  → ranked, explainable candidate report
```

## Scope and data
Uses only public datasets (MoleculeNet: Tox21/BBBP/ESOL, ZINC subset,
PDBbind) under their original licenses — no proprietary or commercially
restricted data. This is a research/educational system demonstrating
modern generative and geometric deep learning for drug discovery, not a
validated tool for real-world lead optimization.

## License
TBD

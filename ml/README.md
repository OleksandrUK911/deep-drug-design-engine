# ml/

Training code and model definitions for the deep learning stack: GNN
property models, SSL pretraining, the generative model, RL optimization,
docking scoring, 3D/equivariant networks, and target conditioning.

Planned layout (populated sprint by sprint per `TODO/ROADMAP.md`):

```
ml/
  data/          # dataset loading, featurization, PyG Dataset/DataLoader
  models/        # GNN, generative, RL, EGNN model definitions
  training/      # training loops, experiment-tracking integration
  evaluation/    # metrics, benchmarks, ablation scripts
  registry/      # model registry / checkpoint metadata
```

Not implemented yet — this is a placeholder for Sprint 2 onward (see
`TODO/ROADMAP.md`).

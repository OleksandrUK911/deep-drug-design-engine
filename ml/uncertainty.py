"""MC Dropout uncertainty estimation (see
TODO/ml/TODO_uncertainty_explainability.md). Requires a model trained with
dropout > 0 (see ml/models/gnn.py) — MC Dropout works by leaving dropout
switched ON at inference time and running multiple stochastic forward
passes; the spread across passes approximates predictive uncertainty.
"""
from __future__ import annotations

import torch
from torch_geometric.data import Data


def _enable_dropout(model: torch.nn.Module) -> None:
    """Set the whole model to eval() (so BatchNorm etc. behave correctly)
    except Dropout layers, which are forced back into train() mode so they
    keep sampling instead of becoming a no-op."""
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_predict(model: torch.nn.Module, data: Data, n_samples: int = 30, device: str = "cpu") -> dict:
    """Run `n_samples` stochastic forward passes on a single graph (or a
    pre-batched set of graphs sharing one `.batch` vector) and return the
    mean prediction and its standard deviation as an uncertainty score.

    For classification models the mean/std are computed in probability
    space (post-sigmoid), which is the quantity that is actually
    meaningful to report alongside a predicted class.
    """
    _enable_dropout(model)
    data = data.to(device)

    samples = []
    for _ in range(n_samples):
        out = model(data.x, data.edge_index, data.batch).squeeze(-1)
        samples.append(out)
    stacked = torch.stack(samples, dim=0)  # [n_samples, batch_size]

    return {
        "mean_logit": stacked.mean(dim=0),
        "std_logit": stacked.std(dim=0),
        "mean_prob": torch.sigmoid(stacked).mean(dim=0),
        "std_prob": torch.sigmoid(stacked).std(dim=0),
        "samples": stacked,
    }

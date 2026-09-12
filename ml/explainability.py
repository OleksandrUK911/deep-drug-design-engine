"""GNNExplainer integration (see TODO/ml/TODO_uncertainty_explainability.md):
identify which atoms/bonds a trained property model relied on most for a
given prediction.

PyG's `Explainer` expects a model called as `model(x, edge_index, ...)`
returning a single graph's prediction; our property models take an extra
`batch` argument, so `_SingleGraphWrapper` fixes `batch` to all-zeros
(one graph) and forwards everything else, which is what `Explainer` needs
for graph-level explanation without changing the trained model itself.
"""
from __future__ import annotations

import torch
from torch_geometric.explain import Explainer, GNNExplainer


class _SingleGraphWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index):
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.model(x, edge_index, batch)


def explain_prediction(model: torch.nn.Module, x, edge_index, epochs: int = 200):
    """Return a PyG `Explanation` object with `.node_mask` (per-atom
    importance, shape [num_nodes, num_features] or [num_nodes]) and
    `.edge_mask` (per-bond importance) for one molecule's prediction.
    """
    wrapped = _SingleGraphWrapper(model)
    explainer = Explainer(
        model=wrapped,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="binary_classification", task_level="graph", return_type="raw"),
    )
    return explainer(x, edge_index)


def atom_importance(explanation) -> torch.Tensor:
    """Collapse the per-feature node mask into one importance score per
    atom (sum of absolute importance across that atom's features) — the
    number the frontend heatmap (TODO/frontend/TODO_molecule_visualization.md)
    actually wants.
    """
    node_mask = explanation.node_mask
    return node_mask.abs().sum(dim=-1)

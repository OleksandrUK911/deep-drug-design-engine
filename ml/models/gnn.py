"""Baseline GNN property-prediction architectures (see
TODO/ml/TODO_gnn_property_model.md): GCN and GIN, both ending in global
mean pooling + an MLP readout head. Kept small and dependency-light
(PyTorch Geometric only) since these are baselines to compare against a
classical RDKit-descriptor + XGBoost model, not the final word.

Both models take a `dropout` argument used two ways: as ordinary
regularization during training, and — left switched on at inference time —
as the sampling distribution for MC Dropout uncertainty estimation (see
ml/uncertainty.py and TODO/ml/TODO_uncertainty_explainability.md).
"""
from __future__ import annotations

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

from ml.models.encoder import GINEncoder


class GCNPropertyModel(nn.Module):
    """Graph Convolutional Network: stacks GCNConv layers, then a
    graph-level MLP head. `out_dim` = 1 for single-task (classification
    logit or regression value), >1 for Tox21-style multi-task heads.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 3, out_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        graph_emb = global_mean_pool(x, batch)
        return self.head(graph_emb)


class GINPropertyModel(nn.Module):
    """Graph Isomorphism Network — provably more expressive than GCN at
    distinguishing non-isomorphic graphs (Xu et al., 2019), used here as
    the second architecture to compare against GCN on the same splits.

    The conv stack is `ml.models.encoder.GINEncoder` so weights pretrained
    by `ml/training/pretrain_ssl.py` load straight into `self.encoder`
    (see `load_pretrained_encoder`).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 3, out_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        self.encoder = GINEncoder(in_dim, hidden_dim, num_layers, dropout=dropout)
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def load_pretrained_encoder(self, state_dict: dict) -> None:
        self.encoder.load_state_dict(state_dict)

    def forward(self, x, edge_index, batch):
        x = self.encoder(x, edge_index)
        graph_emb = global_mean_pool(x, batch)
        return self.head(graph_emb)


def build_model(architecture: str, in_dim: int, hidden_dim: int = 64, num_layers: int = 3, out_dim: int = 1, dropout: float = 0.0) -> nn.Module:
    if architecture == "gcn":
        return GCNPropertyModel(in_dim, hidden_dim, num_layers, out_dim, dropout=dropout)
    if architecture == "gin":
        return GINPropertyModel(in_dim, hidden_dim, num_layers, out_dim, dropout=dropout)
    raise ValueError(f"Unknown architecture: {architecture!r} (expected 'gcn' or 'gin')")

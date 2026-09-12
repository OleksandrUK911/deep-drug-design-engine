"""Shared GIN graph encoder (conv stack only, no pooling/head) used by both
the supervised property model (ml/models/gnn.py) and SSL pretraining
(ml/training/pretrain_ssl.py) — so pretrained weights load directly into
the fine-tuning model without any architecture translation.
"""
from __future__ import annotations

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINConv


class GINEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # node embeddings, [num_nodes, hidden_dim]

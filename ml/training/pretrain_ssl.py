"""Self-supervised masked-attribute pretraining on ZINC (see
TODO/ml/TODO_ssl_pretraining.md): mask a fraction of atoms' input features
and train the GIN encoder + a small classification head to recover which
element each masked atom actually was. The point isn't the pretext
accuracy itself — it's the encoder weights, which get reused by
ml/training/train_property_model.py's --pretrained-encoder flag.

Usage:
    python -m ml.training.pretrain_ssl --dataset data/raw/zinc250k.csv \
        --smiles-col smiles --n-molecules 5000 --epochs 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader

from ml.data.dataset import load_smiles_and_labels
from ml.data.featurizer import ATOM_SYMBOLS, NUM_ATOM_FEATURES, smiles_to_graph
from ml.models.encoder import GINEncoder
from ml.tracking import log_artifact, log_metrics, tracked_run

# Index NUM_ATOM_SYMBOL_CLASSES: one class per known symbol + one "other" class,
# mirroring the "is_other_symbol" flag added in the featurizer's one-hot block.
NUM_ATOM_SYMBOL_CLASSES = len(ATOM_SYMBOLS) + 1
MASK_FRACTION = 0.15


def mask_graph(graph, mask_fraction: float, generator: torch.Generator):
    """Return (masked_graph, mask_indices, target_symbol_idx). The target
    for a masked atom is the argmax over its original one-hot symbol block
    (the first NUM_ATOM_SYMBOL_CLASSES features) — i.e. which element it
    was before masking.
    """
    num_nodes = graph.x.size(0)
    num_mask = max(1, int(num_nodes * mask_fraction))
    perm = torch.randperm(num_nodes, generator=generator)
    mask_idx = perm[:num_mask]

    target = graph.x[mask_idx, :NUM_ATOM_SYMBOL_CLASSES].argmax(dim=-1).clone()

    masked_x = graph.x.clone()
    masked_x[mask_idx] = 0.0  # zero out the whole feature vector -> "unknown atom" token

    masked = graph.clone()
    masked.x = masked_x
    return masked, mask_idx, target


class SSLPretrainModel(nn.Module):
    """Encoder + a per-node linear head predicting the masked atom's
    element class. Only `self.encoder`'s state_dict is reused downstream —
    the head is pretext-task-specific and discarded after pretraining."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.encoder = GINEncoder(in_dim, hidden_dim, num_layers)
        self.mask_head = nn.Linear(hidden_dim, NUM_ATOM_SYMBOL_CLASSES)

    def forward(self, x, edge_index):
        node_emb = self.encoder(x, edge_index)
        return self.mask_head(node_emb)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--n-molecules", type=int, default=5000, help="subsample size for tractable CPU pretraining")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    smiles, _ = load_smiles_and_labels(args.dataset, args.smiles_col, [])
    smiles = smiles[: args.n_molecules]
    print(f"Pretraining on {len(smiles)} molecules (of {args.n_molecules} requested)")

    graphs = [smiles_to_graph(s) for s in smiles]
    graphs = [g for g in graphs if g is not None and g.x.size(0) >= 3]

    model = SSLPretrainModel(NUM_ATOM_FEATURES, args.hidden_dim, args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)

    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=True)

    params = vars(args) | {"mask_fraction": MASK_FRACTION, "n_graphs": len(graphs)}
    with tracked_run("ssl_pretraining", f"masked_atom_{args.dataset.stem}", params, args.dataset):
        for epoch in range(args.epochs):
            model.train()
            total_loss, total_correct, total_masked = 0.0, 0, 0

            for batch in loader:
                # Mask each graph in the batch individually so batch.batch
                # indices stay valid, then re-batch the masked versions.
                masked_list, all_mask_idx, all_targets, offset = [], [], [], 0
                for i in range(batch.num_graphs):
                    g = batch.get_example(i)
                    masked_g, mask_idx, target = mask_graph(g, MASK_FRACTION, generator)
                    masked_list.append(masked_g)
                    all_mask_idx.append(mask_idx + offset)
                    all_targets.append(target)
                    offset += g.x.size(0)

                from torch_geometric.data import Batch as PyGBatch

                remasked_batch = PyGBatch.from_data_list(masked_list).to(device)
                mask_idx = torch.cat(all_mask_idx).to(device)
                targets = torch.cat(all_targets).to(device)

                optimizer.zero_grad()
                logits = model(remasked_batch.x, remasked_batch.edge_index)
                masked_logits = logits[mask_idx]
                loss = F.cross_entropy(masked_logits, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * targets.size(0)
                total_correct += (masked_logits.argmax(dim=-1) == targets).sum().item()
                total_masked += targets.size(0)

            avg_loss = total_loss / max(total_masked, 1)
            accuracy = total_correct / max(total_masked, 1)
            log_metrics({"pretrain_loss": avg_loss, "pretrain_mask_accuracy": accuracy}, step=epoch)
            print(f"epoch {epoch:3d} | masked_atom_loss={avg_loss:.4f} | masked_atom_accuracy={accuracy:.3f}")

        checkpoint_dir = Path("mlruns_checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        ckpt_path = checkpoint_dir / "ssl_encoder.pt"
        torch.save(model.encoder.state_dict(), ckpt_path)
        log_artifact(ckpt_path)
        print(f"Saved pretrained encoder to {ckpt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

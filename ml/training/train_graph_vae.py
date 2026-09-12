"""Train the fixed-size Graph VAE on a ZINC subset, with linear KL
annealing to reduce posterior collapse risk. See
TODO/ml/TODO_generative_model.md.

Usage:
    python -m ml.training.train_graph_vae --dataset data/raw/zinc250k.csv \
        --n-molecules 5000 --epochs 30 --n-samples 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from rdkit import Chem
from torch_geometric.loader import DataLoader

from ml.data.dataset import load_smiles_and_labels
from ml.data.featurizer import NUM_ATOM_FEATURES, canonical_smiles, smiles_to_graph
from ml.generative.graph_vae import GraphVAE, mol_to_fixed_size_targets, vae_loss
from ml.generative.metrics import generation_metrics, internal_diversity
from ml.tracking import log_artifact, log_metrics, tracked_run


def build_training_examples(smiles_list: list[str]):
    """Featurize + compute fixed-size targets, dropping molecules that
    don't fit MAX_ATOMS or use an unsupported element (see
    graph_vae.mol_to_fixed_size_targets)."""
    examples = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        targets = mol_to_fixed_size_targets(mol)
        if targets is None:
            continue
        graph = smiles_to_graph(s)
        if graph is None:
            continue
        atom_types, edge_types = targets
        graph.atom_types = atom_types.unsqueeze(0)
        graph.edge_types = edge_types.unsqueeze(0)
        examples.append(graph)
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--n-molecules", type=int, default=5000)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=56)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--kl-anneal-epochs", type=int, default=15, help="epochs to linearly ramp KL weight 0 -> 1")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    smiles, _ = load_smiles_and_labels(args.dataset, args.smiles_col, [])
    smiles = smiles[: args.n_molecules]
    training_canon = {canonical_smiles(s) for s in smiles} - {None}

    examples = build_training_examples(smiles)
    print(f"Usable training examples: {len(examples)} / {len(smiles)} requested "
          f"(dropped: too many atoms or unsupported element)")

    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True)

    model = GraphVAE(NUM_ATOM_FEATURES, args.hidden_dim, args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    params = vars(args) | {"n_usable_examples": len(examples)}
    with tracked_run("generative_graph_vae", f"graphvae_{args.dataset.stem}", params, args.dataset):
        for epoch in range(args.epochs):
            model.train()
            kl_weight = min(1.0, epoch / max(args.kl_anneal_epochs, 1))
            total_loss, total_atom, total_edge, total_kl, n_batches = 0.0, 0.0, 0.0, 0.0, 0

            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                atom_logits, edge_logits, mu, logvar = model(batch.x, batch.edge_index, batch.batch)
                loss, parts = vae_loss(atom_logits, edge_logits, batch.atom_types, batch.edge_types, mu, logvar, kl_weight)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_atom += parts["atom_loss"]
                total_edge += parts["edge_loss"]
                total_kl += parts["kl"]
                n_batches += 1

            log_metrics({
                "train_loss": total_loss / n_batches,
                "atom_loss": total_atom / n_batches,
                "edge_loss": total_edge / n_batches,
                "kl": total_kl / n_batches,
                "kl_weight": kl_weight,
            }, step=epoch)
            print(f"epoch {epoch:3d} | loss={total_loss/n_batches:.4f} atom={total_atom/n_batches:.4f} "
                  f"edge={total_edge/n_batches:.4f} kl={total_kl/n_batches:.4f} (weight={kl_weight:.2f})")

        samples = model.sample(args.n_samples, device=device)
        samples = [s for s in samples if s is not None]  # tensors_to_mol already returns None for invalid
        # generation_metrics expects raw generated strings including failures counted in validity;
        # reconstruct that by passing a placeholder for the Nones.
        n_decoded_none = args.n_samples - len(samples)
        all_outputs = samples + ["INVALID"] * n_decoded_none
        metrics = generation_metrics(all_outputs, training_canon)
        metrics["internal_diversity"] = internal_diversity(samples[:100]) if samples else 0.0

        print(f"\nGeneration metrics ({args.n_samples} samples): {metrics}")
        log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

        checkpoint_dir = Path("mlruns_checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        ckpt_path = checkpoint_dir / "graph_vae.pt"
        torch.save(model.state_dict(), ckpt_path)
        log_artifact(ckpt_path)

        print("\nSample generated molecules (first 10 valid):")
        for s in samples[:10]:
            print(f"  {s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

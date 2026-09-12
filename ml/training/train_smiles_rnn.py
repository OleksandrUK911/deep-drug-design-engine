"""Train the char-level LSTM SMILES generator on a ZINC subset and report
validity/uniqueness/novelty of sampled molecules. See
TODO/ml/TODO_generative_model.md and TODO/ml/TODO_evaluation_metrics.md.

Usage:
    python -m ml.training.train_smiles_rnn --dataset data/raw/zinc250k.csv \
        --n-molecules 20000 --epochs 10 --n-samples 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from ml.data.dataset import load_smiles_and_labels
from ml.data.featurizer import canonical_smiles
from ml.generative.metrics import generation_metrics, internal_diversity
from ml.generative.smiles_rnn import CharRNNGenerator
from ml.generative.smiles_vocab import SmilesVocab
from ml.tracking import log_artifact, log_metrics, tracked_run


class SmilesDataset(Dataset):
    def __init__(self, smiles_list: list[str], vocab: SmilesVocab):
        self.encoded = [torch.tensor(vocab.encode(s), dtype=torch.long) for s in smiles_list]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx]


def collate(batch: list[torch.Tensor], pad_idx: int) -> torch.Tensor:
    return pad_sequence(batch, batch_first=True, padding_value=pad_idx)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--n-molecules", type=int, default=20000)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    smiles, _ = load_smiles_and_labels(args.dataset, args.smiles_col, [])
    smiles = smiles[: args.n_molecules]
    training_canon = {canonical_smiles(s) for s in smiles} - {None}
    print(f"Training on {len(smiles)} molecules")

    vocab = SmilesVocab(smiles)
    print(f"Vocab size: {len(vocab)}")

    dataset = SmilesDataset(smiles, vocab)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, vocab.pad_idx))

    model = CharRNNGenerator(len(vocab), hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    params = vars(args) | {"vocab_size": len(vocab)}
    with tracked_run("generative_smiles_rnn", f"charrnn_{args.dataset.stem}", params, args.dataset):
        for epoch in range(args.epochs):
            model.train()
            total_loss, total_tokens = 0.0, 0
            for batch in loader:
                batch = batch.to(device)
                inputs, targets = batch[:, :-1], batch[:, 1:]

                optimizer.zero_grad()
                logits, _ = model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=vocab.pad_idx
                )
                loss.backward()
                optimizer.step()

                n_tokens = (targets != vocab.pad_idx).sum().item()
                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens

            avg_loss = total_loss / max(total_tokens, 1)
            log_metrics({"train_loss": avg_loss}, step=epoch)
            print(f"epoch {epoch:3d} | loss={avg_loss:.4f}")

        samples = model.sample(vocab, args.n_samples, temperature=args.temperature, device=device)
        metrics = generation_metrics(samples, training_canon)
        valid_samples = [canonical_smiles(s) for s in samples]
        valid_samples = [s for s in valid_samples if s is not None]
        metrics["internal_diversity"] = internal_diversity(valid_samples[:100])

        print(f"Generation metrics ({args.n_samples} samples): {metrics}")
        log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

        checkpoint_dir = Path("mlruns_checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        ckpt_path = checkpoint_dir / "smiles_rnn.pt"
        torch.save({"state_dict": model.state_dict(), "vocab_itos": vocab.itos}, ckpt_path)
        log_artifact(ckpt_path)

        print("\nSample generated molecules (first 10 valid):")
        for s in valid_samples[:10]:
            print(f"  {s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

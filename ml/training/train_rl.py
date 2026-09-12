"""RL fine-tuning of the pretrained char-RNN generator against a
composite reward (property + QED + synthesizability [+ docking]). See
TODO/ml/TODO_rl_optimization.md.

Docking is off by default: scoring every molecule in every RL batch
against Vina (~1s/molecule) would make even a short run prohibitively
slow (batch_size=64 x n_steps=60 => 3840 docking calls). It can be
enabled for a short, separate run to validate the full composite reward
end-to-end (see the --with-docking flag and ml/evaluation/RESULTS.md for
that smaller-scale experiment) without being the default training mode.

Usage:
    python -m ml.training.train_rl --smiles-checkpoint mlruns_checkpoints/smiles_rnn.pt \
        --property-checkpoint mlruns_checkpoints/gin_BBBP.pt --n-steps 60 --batch-size 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from ml.data.featurizer import NUM_ATOM_FEATURES, canonical_smiles
from ml.generative.metrics import generation_metrics, internal_diversity
from ml.generative.smiles_rnn import CharRNNGenerator
from ml.generative.smiles_vocab import SmilesVocab
from ml.models.gnn import build_model
from ml.rl.reinvent import train_reinvent
from ml.rl.reward import DEFAULT_WEIGHTS_NO_DOCKING, DEFAULT_WEIGHTS_WITH_DOCKING, RewardConfig, score_batch
from ml.tracking import log_artifact, log_metrics, tracked_run


def load_pretrained_generator(checkpoint_path: Path, device: str) -> tuple[CharRNNGenerator, SmilesVocab]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    vocab = SmilesVocab.__new__(SmilesVocab)
    vocab.itos = ckpt["vocab_itos"]
    vocab.stoi = {c: i for i, c in enumerate(vocab.itos)}
    vocab.pad_idx, vocab.bos_idx, vocab.eos_idx = vocab.stoi["<pad>"], vocab.stoi["<bos>"], vocab.stoi["<eos>"]

    model = CharRNNGenerator(len(vocab))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device), vocab


def plot_reward_distributions(pre_rewards: list[float], post_rewards: list[float], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = [i / 20 for i in range(21)]
    ax.hist(pre_rewards, bins=bins, alpha=0.6, label=f"Before RL (mean={sum(pre_rewards)/len(pre_rewards):.3f})")
    ax.hist(post_rewards, bins=bins, alpha=0.6, label=f"After RL (mean={sum(post_rewards)/len(post_rewards):.3f})")
    ax.set_xlabel("Composite reward")
    ax.set_ylabel("Count")
    ax.set_title("Reward distribution before vs. after REINVENT fine-tuning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles-checkpoint", required=True, type=Path)
    parser.add_argument("--property-checkpoint", type=Path, default=None)
    parser.add_argument("--property-hidden-dim", type=int, default=64)
    parser.add_argument("--property-num-layers", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=60.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-eval-samples", type=int, default=200)
    parser.add_argument("--with-docking", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pretrained_model, vocab = load_pretrained_generator(args.smiles_checkpoint, device)

    property_model = None
    if args.property_checkpoint is not None:
        property_model = build_model("gin", in_dim=NUM_ATOM_FEATURES, hidden_dim=args.property_hidden_dim, num_layers=args.property_num_layers)
        property_model.load_state_dict(torch.load(args.property_checkpoint, map_location=device))
        property_model.to(device)

    docking_fn = None
    weights = dict(DEFAULT_WEIGHTS_NO_DOCKING)
    if args.with_docking:
        from ml.docking.vina_wrapper import VINA_EXE, dock

        receptor_pdbqt = Path("data/fixtures/docking/split/receptor.pdbqt")
        if not VINA_EXE.exists() or not receptor_pdbqt.exists():
            print("--with-docking requested but Vina/receptor not available; falling back to no-docking reward.")
        else:
            center = (-1.7591111111111113, 14.461, 16.915777777777777)

            def docking_fn(smiles: str):
                try:
                    result = dock(smiles, receptor_pdbqt, center, exhaustiveness=4, use_cache=True)
                    return result["best_affinity_kcal_mol"]
                except Exception:
                    return None

            weights = dict(DEFAULT_WEIGHTS_WITH_DOCKING)

    reward_config = RewardConfig(weights=weights, property_model=property_model, device=device, docking_fn=docking_fn)
    print(f"Reward weights: {weights}")

    # Baseline: reward distribution of the pretrained model, before any RL.
    with torch.no_grad():
        pre_samples = pretrained_model.sample(vocab, args.n_eval_samples, device=device)
    pre_rewards = score_batch(pre_samples, reward_config)
    pre_mean_reward = sum(r["total"] for r in pre_rewards) / len(pre_rewards)
    pre_valid = [canonical_smiles(s) for s in pre_samples]
    pre_valid = [s for s in pre_valid if s is not None]
    pre_diversity = internal_diversity(pre_valid[:100]) if pre_valid else 0.0
    print(f"BEFORE RL: mean_reward={pre_mean_reward:.4f} validity={len(pre_valid)/len(pre_samples):.2f} diversity={pre_diversity:.4f}")

    params = vars(args) | {"weights": weights}
    with tracked_run("rl_reinvent", f"reinvent_{args.smiles_checkpoint.stem}", params, args.smiles_checkpoint):
        log_metrics({"pre_rl_mean_reward": pre_mean_reward, "pre_rl_validity": len(pre_valid) / len(pre_samples), "pre_rl_diversity": pre_diversity})

        agent = train_reinvent(
            pretrained_model, vocab, reward_config,
            n_steps=args.n_steps, batch_size=args.batch_size, sigma=args.sigma, lr=args.lr, device=device,
            callback=lambda record, *_: log_metrics(
                {"rl_loss": record["loss"], "rl_mean_reward": record["mean_reward"],
                 "rl_valid_frac": record["valid_frac"], "rl_unique_frac": record["unique_frac"]},
                step=record["step"],
            ),
        )

        with torch.no_grad():
            post_samples = agent.sample(vocab, args.n_eval_samples, device=device)
        post_rewards = score_batch(post_samples, reward_config)
        post_mean_reward = sum(r["total"] for r in post_rewards) / len(post_rewards)
        post_valid = [canonical_smiles(s) for s in post_samples]
        post_valid = [s for s in post_valid if s is not None]
        post_diversity = internal_diversity(post_valid[:100]) if post_valid else 0.0

        print(f"AFTER RL:  mean_reward={post_mean_reward:.4f} validity={len(post_valid)/len(post_samples):.2f} diversity={post_diversity:.4f}")
        print(f"Reward delta: {post_mean_reward - pre_mean_reward:+.4f}")

        gen_metrics = generation_metrics(post_samples, training_smiles=set())
        log_metrics({
            "post_rl_mean_reward": post_mean_reward, "post_rl_validity": len(post_valid) / len(post_samples),
            "post_rl_diversity": post_diversity, "reward_delta": post_mean_reward - pre_mean_reward,
            **{k: v for k, v in gen_metrics.items() if isinstance(v, (int, float))},
        })

        checkpoint_dir = Path("mlruns_checkpoints")
        ckpt_path = checkpoint_dir / "rl_agent.pt"
        torch.save({"state_dict": agent.state_dict(), "vocab_itos": vocab.itos}, ckpt_path)
        log_artifact(ckpt_path)

        plot_path = checkpoint_dir / "reward_distribution.png"
        plot_reward_distributions([r["total"] for r in pre_rewards], [r["total"] for r in post_rewards], plot_path)
        log_artifact(plot_path)

        print("\nTop 10 post-RL molecules by reward:")
        ranked = sorted(zip(post_samples, post_rewards), key=lambda x: -x[1]["total"])
        for smiles, r in ranked[:10]:
            canon = canonical_smiles(smiles)
            if canon:
                print(f"  reward={r['total']:.3f} {r}  {canon}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Multi-objective (Pareto) analysis of the RL agent's generated
candidates, compared against the single-scalarized-reward approach from
Sprint 6. See TODO/ml/TODO_multi_objective_optimization.md.

Usage:
    python -m ml.training.analyze_pareto --agent-checkpoint mlruns_checkpoints/rl_agent.pt \
        --property-checkpoint mlruns_checkpoints/gin_BBBP.pt \
        --training-dataset data/raw/zinc250k.csv --n-samples 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from ml.data.dataset import load_smiles_and_labels
from ml.data.featurizer import NUM_ATOM_FEATURES, canonical_smiles
from ml.generative.smiles_rnn import CharRNNGenerator
from ml.generative.smiles_vocab import SmilesVocab
from ml.models.gnn import build_model
from ml.rl.pareto import compute_objectives, non_dominated_front, pareto_knee_points, scalarization_reachable_set
from ml.rl.reward import DEFAULT_WEIGHTS_NO_DOCKING, RewardConfig, score_batch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-checkpoint", required=True, type=Path)
    parser.add_argument("--property-checkpoint", required=True, type=Path)
    parser.add_argument("--training-dataset", required=True, type=Path)
    parser.add_argument("--n-training-ref", type=int, default=2000, help="training molecules used as the novelty reference set")
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--grid-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cpu"

    ckpt = torch.load(args.agent_checkpoint, map_location=device)
    vocab = SmilesVocab.__new__(SmilesVocab)
    vocab.itos = ckpt["vocab_itos"]
    vocab.stoi = {c: i for i, c in enumerate(vocab.itos)}
    vocab.pad_idx, vocab.bos_idx, vocab.eos_idx = vocab.stoi["<pad>"], vocab.stoi["<bos>"], vocab.stoi["<eos>"]
    agent = CharRNNGenerator(len(vocab))
    agent.load_state_dict(ckpt["state_dict"])

    property_model = build_model("gin", in_dim=NUM_ATOM_FEATURES, hidden_dim=64, num_layers=3)
    property_model.load_state_dict(torch.load(args.property_checkpoint, map_location=device))

    training_smiles, _ = load_smiles_and_labels(args.training_dataset, "smiles", [])
    training_ref = training_smiles[: args.n_training_ref]

    print(f"Sampling {args.n_samples} candidates from the RL agent...")
    with torch.no_grad():
        samples = agent.sample(vocab, args.n_samples, device=device)
    samples = [s for s in samples if canonical_smiles(s) is not None]
    print(f"Valid unique candidates: {len(set(canonical_smiles(s) for s in samples))}/{len(samples)}")

    reward_config = RewardConfig(weights=DEFAULT_WEIGHTS_NO_DOCKING, property_model=property_model, device=device)

    print("Computing objective vectors (property, synth, novelty)...")
    objectives = compute_objectives(samples, reward_config, training_ref, include_docking=False)
    print(f"Objectives shape: {objectives.values.shape} ({objectives.names})")

    front_idx = non_dominated_front(objectives.values)
    print(f"\nPareto front size: {len(front_idx)} / {len(objectives.smiles)}")

    reachable = scalarization_reachable_set(objectives.values, n_grid_steps=args.grid_steps)
    print(f"Scalarization-reachable set size (grid of {args.grid_steps} steps per axis): {len(reachable)}")

    front_set = set(front_idx.tolist())
    unreachable_pareto = front_set - reachable
    print(f"Pareto-optimal candidates UNREACHABLE by any fixed weight vector on the grid: {len(unreachable_pareto)}")

    print("\n--- Comparison: single-weighted-reward top pick vs. Pareto-only candidates ---")
    rewards = score_batch(objectives.smiles, reward_config)
    best_single_idx = max(range(len(rewards)), key=lambda i: rewards[i]["total"])
    print(f"Best single-scalarized-reward candidate: {objectives.smiles[best_single_idx]}")
    print(f"  objectives: {dict(zip(objectives.names, objectives.values[best_single_idx].round(3)))}")

    if unreachable_pareto:
        for idx in list(unreachable_pareto)[:3]:
            print(f"Pareto-only candidate (never a fixed-weight argmax): {objectives.smiles[idx]}")
            print(f"  objectives: {dict(zip(objectives.names, objectives.values[idx].round(3)))}")

    knees = pareto_knee_points(objectives.values, front_idx, n_knees=3)
    print("\n--- Pareto front 'knee' points (balanced trade-off candidates) ---")
    for idx in knees:
        print(f"  {objectives.smiles[idx]}  {dict(zip(objectives.names, objectives.values[idx].round(3)))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

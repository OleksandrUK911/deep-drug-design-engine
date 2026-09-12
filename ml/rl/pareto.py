"""Multi-objective (Pareto) analysis over a pool of candidates (see
TODO/ml/TODO_multi_objective_optimization.md) — an explicit alternative
to the single scalarized reward in ml/rl/reward.py, which hides the real
trade-off between potency, synthesizability and novelty behind one
number.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from ml.rl.reward import RewardConfig, normalize_sascore
from ml.synthesizability import sa_score


@dataclass
class ObjectiveSet:
    names: list[str]
    values: np.ndarray  # [n_candidates, n_objectives], all "higher is better"
    smiles: list[str]


def _nearest_neighbor_novelty(smiles: str, training_fps: list) -> float:
    """1 - max Tanimoto similarity to any training molecule -- a
    continuous novelty score (0 = identical to something in training,
    1 = maximally dissimilar to the whole training set), which is far
    more informative as a Pareto axis than a binary in/not-in-training
    flag.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not training_fps:
        return 0.0
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    sims = DataStructs.BulkTanimotoSimilarity(fp, training_fps)
    return 1.0 - max(sims)


def compute_objectives(
    smiles_list: list[str],
    reward_config: RewardConfig,
    training_smiles: list[str],
    include_docking: bool = False,
) -> ObjectiveSet:
    """Property (from the trained GNN) + SAScore-synthesizability +
    nearest-neighbor novelty vs. training, optionally + docking -- as an
    explicit vector rather than one scalarized number.
    """
    training_mols = [Chem.MolFromSmiles(s) for s in training_smiles]
    training_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in training_mols if m is not None]

    names = ["property", "synth", "novelty"]
    if include_docking:
        names.append("docking")

    rows, valid_smiles = [], []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        from torch_geometric.data import Batch as PyGBatch

        from ml.data.featurizer import smiles_to_graph
        import torch

        property_score = 0.0
        if reward_config.property_model is not None:
            graph = smiles_to_graph(smiles)
            if graph is not None:
                with torch.no_grad():
                    batch = PyGBatch.from_data_list([graph]).to(reward_config.device)
                    logit = reward_config.property_model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
                    property_score = torch.sigmoid(logit).item()

        sa = sa_score(smiles)
        synth_score = normalize_sascore(sa) if sa is not None else 0.0
        novelty_score = _nearest_neighbor_novelty(smiles, training_fps)

        row = [property_score, synth_score, novelty_score]
        if include_docking:
            affinity = reward_config.docking_fn(smiles) if reward_config.docking_fn else None
            from ml.rl.reward import normalize_docking

            row.append(normalize_docking(affinity) if affinity is not None else 0.0)

        rows.append(row)
        valid_smiles.append(smiles)

    return ObjectiveSet(names=names, values=np.array(rows), smiles=valid_smiles)


def non_dominated_front(values: np.ndarray) -> np.ndarray:
    """Indices of the Pareto-optimal (rank-0, non-dominated) points.
    `values`: [n, k], all objectives "higher is better". O(n^2) — fine
    for candidate pools up to a few thousand, which is the scale this
    project operates at.
    """
    n = values.shape[0]
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is >= i on every objective and > on at least one.
            if np.all(values[j] >= values[i]) and np.any(values[j] > values[i]):
                is_dominated[i] = True
                break
    return np.where(~is_dominated)[0]


def scalarization_reachable_set(values: np.ndarray, n_grid_steps: int = 10) -> set[int]:
    """Indices of points that are the argmax for *some* fixed non-negative
    weight vector (summing to 1) on a simplex grid. Points on a concave
    (non-convex) region of the true Pareto front can be Pareto-optimal yet
    never be the argmax of any linear scalarization — this function's
    complement of `non_dominated_front` is exactly how we demonstrate that.
    """
    k = values.shape[1]
    reachable: set[int] = set()

    # Grid of weight vectors on the (k-1)-simplex, step size 1/n_grid_steps.
    for combo in itertools.product(range(n_grid_steps + 1), repeat=k):
        if sum(combo) != n_grid_steps:
            continue
        weights = np.array(combo) / n_grid_steps
        scores = values @ weights
        reachable.add(int(np.argmax(scores)))

    return reachable


def pareto_knee_points(values: np.ndarray, front_indices: np.ndarray, n_knees: int = 3) -> np.ndarray:
    """Representative "knee" points of the Pareto front: normalize each
    objective to [0,1] over the front, then pick the front members closest
    to the ideal point (1,1,...,1) by Euclidean distance -- a standard,
    simple way to pick a handful of "balanced" candidates out of a
    (potentially large) front for a report, rather than showing all of
    them undifferentiated.
    """
    front_values = values[front_indices]
    mins, maxs = front_values.min(axis=0), front_values.max(axis=0)
    span = np.where(maxs - mins > 1e-9, maxs - mins, 1.0)
    normalized = (front_values - mins) / span

    distances = np.linalg.norm(normalized - 1.0, axis=1)
    order = np.argsort(distances)[:n_knees]
    return front_indices[order]

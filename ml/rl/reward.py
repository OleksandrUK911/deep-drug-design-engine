"""Composite reward for RL-optimized generation (see
TODO/ml/TODO_rl_optimization.md): predicted property (from the trained
GNN) + QED drug-likeness + synthesizability (SAScore) [+ optional docking
score], each normalized to roughly [0, 1] and weighted so no single term
dominates training. Invalid molecules get reward 0 on every component —
no extra penalty term needed since 0 is already the floor of every
normalized component.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from rdkit import Chem
from rdkit.Chem import QED
from torch_geometric.data import Batch

from ml.data.featurizer import smiles_to_graph
from ml.synthesizability import sa_score

DEFAULT_WEIGHTS_NO_DOCKING = {"property": 0.45, "qed": 0.25, "synth": 0.30}
DEFAULT_WEIGHTS_WITH_DOCKING = {"property": 0.35, "qed": 0.15, "synth": 0.20, "docking": 0.30}


def normalize_sascore(score: float) -> float:
    """SAScore in [1, 10] (1=easy) -> [0, 1] (1=easy, higher=better),
    clipped so a rare out-of-range value can't blow up the composite."""
    return max(0.0, min(1.0, (10.0 - score) / 9.0))


def normalize_docking(affinity_kcal_mol: float, best: float = -12.0, worst: float = 0.0) -> float:
    """Vina affinity (kcal/mol, more negative = better) -> [0, 1]."""
    clipped = max(min(affinity_kcal_mol, worst), best)
    return (worst - clipped) / (worst - best)


@dataclass
class RewardConfig:
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS_NO_DOCKING))
    property_model: Optional[torch.nn.Module] = None
    device: str = "cpu"
    docking_fn: Optional[Callable[[str], Optional[float]]] = None  # smiles -> affinity kcal/mol, or None if it failed


@torch.no_grad()
def _predict_property(smiles: str, model: torch.nn.Module, device: str) -> Optional[float]:
    graph = smiles_to_graph(smiles)
    if graph is None:
        return None
    batch = Batch.from_data_list([graph]).to(device)
    model.eval()
    logit = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
    return torch.sigmoid(logit).item()


def score_molecule(smiles: str, config: RewardConfig) -> dict:
    """Return every reward component plus the weighted total for one
    molecule. Always returns all keys in config.weights (0.0 if that
    component couldn't be computed) so callers can log a consistent
    breakdown across a whole batch, valid or not."""
    components = {k: 0.0 for k in config.weights}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {**components, "total": 0.0, "valid": False}

    if "qed" in config.weights:
        try:
            components["qed"] = QED.qed(mol)
        except Exception:
            components["qed"] = 0.0

    if "synth" in config.weights:
        sa = sa_score(smiles)
        components["synth"] = normalize_sascore(sa) if sa is not None else 0.0

    if "property" in config.weights and config.property_model is not None:
        prop = _predict_property(smiles, config.property_model, config.device)
        components["property"] = prop if prop is not None else 0.0

    if "docking" in config.weights and config.docking_fn is not None:
        affinity = config.docking_fn(smiles)
        components["docking"] = normalize_docking(affinity) if affinity is not None else 0.0

    total = sum(config.weights[k] * components[k] for k in config.weights)
    return {**components, "total": total, "valid": True}


def score_batch(smiles_list: list[str], config: RewardConfig) -> list[dict]:
    return [score_molecule(s, config) for s in smiles_list]

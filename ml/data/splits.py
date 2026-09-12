"""Deduplication and scaffold splitting (see
TODO/data/TODO_quality_validation.md). A random split leaks information
because near-duplicate scaffolds land in both train and test; scaffold
split groups molecules by Bemis-Murcko scaffold first, so structurally
related molecules stay together on one side of the split.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .featurizer import canonical_smiles


def murcko_scaffold(smiles: str) -> Optional[str]:
    """Bemis-Murcko generic scaffold SMILES for a molecule, or None if the
    input SMILES is invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold_mol, canonical=True)


def dedup_smiles(smiles_list: list[str]) -> list[int]:
    """Return the indices of `smiles_list` to keep after removing duplicates
    by canonical SMILES (first occurrence wins). Invalid SMILES are dropped
    entirely — validity filtering happens here, not just in the featurizer,
    so downstream split/training code never sees them."""
    seen: set[str] = set()
    keep: list[int] = []
    for idx, smiles in enumerate(smiles_list):
        canon = canonical_smiles(smiles)
        if canon is None or canon in seen:
            continue
        seen.add(canon)
        keep.append(idx)
    return keep


def scaffold_split(
    smiles_list: list[str],
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 0,
) -> tuple[list[int], list[int], list[int]]:
    """DeepChem-style scaffold split: group molecule indices by Murcko
    scaffold, sort groups largest-first (deterministically, with `seed`
    breaking ties among equal-size groups), then greedily fill train, then
    valid, then test. Molecules with an invalid/unparseable SMILES are
    excluded from all three splits.
    """
    if abs(frac_train + frac_valid + frac_test - 1.0) > 1e-6:
        raise ValueError("split fractions must sum to 1.0")

    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold = murcko_scaffold(smiles)
        if scaffold is None:
            continue
        scaffold_to_indices[scaffold].append(idx)

    rng_key = lambda item: (-len(item[1]), hash((item[0], seed)))
    ordered_groups = sorted(scaffold_to_indices.items(), key=rng_key)

    n_total = sum(len(v) for _, v in ordered_groups)
    n_train_cutoff = frac_train * n_total
    n_valid_cutoff = (frac_train + frac_valid) * n_total

    train_idx: list[int] = []
    valid_idx: list[int] = []
    test_idx: list[int] = []
    for _, indices in ordered_groups:
        if len(train_idx) + len(indices) <= n_train_cutoff or not train_idx:
            train_idx.extend(indices)
        elif len(train_idx) + len(valid_idx) + len(indices) <= n_valid_cutoff or not valid_idx:
            valid_idx.extend(indices)
        else:
            test_idx.extend(indices)

    return train_idx, valid_idx, test_idx

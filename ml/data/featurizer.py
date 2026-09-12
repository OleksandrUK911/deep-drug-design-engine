"""SMILES -> molecular graph featurization shared by the GNN property model
and the generative model (see TODO/data/TODO_preprocessing_pipeline.md).

Kept dependency-light (RDKit + torch only) so it can be unit-tested without
pulling in the rest of the training stack.
"""
from __future__ import annotations

from typing import Optional

import torch
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data

# Common elements in drug-like molecules (MoleculeNet/ZINC). Anything else
# falls back to the "other" slot rather than raising, so featurization never
# crashes on a single unusual atom.
ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si"]
HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]

NUM_ATOM_FEATURES = len(ATOM_SYMBOLS) + 1 + len(HYBRIDIZATIONS) + 1 + 5 + 3
NUM_BOND_FEATURES = len(BOND_TYPES) + 2


def _one_hot(value, choices) -> list[float]:
    return [1.0 if value == c else 0.0 for c in choices]


def canonical_smiles(smiles: str) -> Optional[str]:
    """Parse + sanitize + re-serialize a SMILES string. Returns None if RDKit
    cannot parse or sanitize it (invalid molecule)."""
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def atom_features(atom: rdchem.Atom) -> list[float]:
    symbol_oh = _one_hot(atom.GetSymbol(), ATOM_SYMBOLS)
    is_other_symbol = [1.0 if atom.GetSymbol() not in ATOM_SYMBOLS else 0.0]
    hybridization_oh = _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
    is_other_hybridization = [
        1.0 if atom.GetHybridization() not in HYBRIDIZATIONS else 0.0
    ]
    degree_oh = _one_hot(min(atom.GetDegree(), 4), [0, 1, 2, 3, 4])
    misc = [
        float(atom.GetFormalCharge()),
        1.0 if atom.GetIsAromatic() else 0.0,
        float(atom.GetTotalNumHs()),
    ]
    return symbol_oh + is_other_symbol + hybridization_oh + is_other_hybridization + degree_oh + misc


def bond_features(bond: rdchem.Bond) -> list[float]:
    bond_type_oh = _one_hot(bond.GetBondType(), BOND_TYPES)
    misc = [
        1.0 if bond.GetIsConjugated() else 0.0,
        1.0 if bond.IsInRing() else 0.0,
    ]
    return bond_type_oh + misc


def mol_to_graph(mol: rdchem.Mol, y: Optional[float] = None) -> Data:
    """Convert an RDKit Mol into a PyTorch Geometric Data object with
    per-atom node features and per-bond (undirected, duplicated both
    directions) edge features."""
    xs = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(xs, dtype=torch.float)

    edge_indices: list[list[int]] = []
    edge_attrs: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = bond_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [feats, feats]

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        # Single-atom molecule: no bonds, but keep tensor shapes consistent.
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, NUM_BOND_FEATURES), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)
    return data


def smiles_to_graph(smiles: str, y: Optional[float] = None) -> Optional[Data]:
    """End-to-end SMILES -> Data. Returns None for anything RDKit rejects,
    so callers can filter invalid rows out of a dataset (see
    TODO/data/TODO_quality_validation.md)."""
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        return None
    return mol_to_graph(mol, y=y)

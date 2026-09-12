import pytest

from ml.data.featurizer import (
    NUM_ATOM_FEATURES,
    NUM_BOND_FEATURES,
    canonical_smiles,
    smiles_to_graph,
)


def test_canonical_smiles_valid():
    assert canonical_smiles("c1ccccc1") == "c1ccccc1"


def test_canonical_smiles_invalid_returns_none():
    assert canonical_smiles("not_a_smiles(((") is None


def test_canonical_smiles_normalizes_equivalent_forms():
    # Two ways of writing benzene should canonicalize to the same string.
    a = canonical_smiles("C1=CC=CC=C1")
    b = canonical_smiles("c1ccccc1")
    assert a == b


def test_smiles_to_graph_shapes():
    data = smiles_to_graph("CCO")  # ethanol: 3 heavy atoms, 2 bonds
    assert data is not None
    assert data.x.shape == (3, NUM_ATOM_FEATURES)
    assert data.edge_index.shape == (2, 4)  # 2 bonds x 2 directions
    assert data.edge_attr.shape == (4, NUM_BOND_FEATURES)


def test_smiles_to_graph_invalid_returns_none():
    assert smiles_to_graph("this is not valid") is None


def test_smiles_to_graph_single_atom_no_bonds():
    data = smiles_to_graph("[Ar]")
    assert data is not None
    assert data.x.shape[0] == 1
    assert data.edge_index.shape == (2, 0)


def test_smiles_to_graph_with_label():
    data = smiles_to_graph("CCO", y=1.23)
    assert data.y.item() == pytest.approx(1.23)

"""End-to-end sanity checks against the small committed fixture samples
(data/fixtures/*.csv) — not the full datasets, so this runs fast and
without network access, but exercises the real CSV -> graph pipeline.
"""
from pathlib import Path

from ml.data.dataset import load_smiles_and_labels
from ml.data.featurizer import smiles_to_graph
from ml.data.splits import dedup_smiles, scaffold_split

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "fixtures"


def test_bbbp_fixture_loads_and_featurizes():
    smiles, labels = load_smiles_and_labels(FIXTURES / "bbbp_sample.csv", "smiles", ["p_np"])
    assert len(smiles) > 0
    assert len(smiles) == len(labels)

    graphs = [smiles_to_graph(s, y=float(labels.iloc[i]["p_np"])) for i, s in enumerate(smiles)]
    assert all(g is not None for g in graphs)


def test_esol_fixture_loads_and_featurizes():
    label_col = "measured log solubility in mols per litre"
    smiles, labels = load_smiles_and_labels(FIXTURES / "esol_sample.csv", "smiles", [label_col])
    assert len(smiles) > 0

    graphs = [smiles_to_graph(s) for s in smiles]
    assert all(g is not None for g in graphs)


def test_zinc250k_fixture_dedup_and_scaffold_split():
    smiles, _ = load_smiles_and_labels(FIXTURES / "zinc250k_sample.csv", "smiles", ["logP"])
    keep = dedup_smiles(smiles)
    assert len(keep) <= len(smiles)
    assert len(keep) > 0

    deduped = [smiles[i] for i in keep]
    train, valid, test = scaffold_split(deduped, frac_train=0.7, frac_valid=0.15, frac_test=0.15)
    assert sorted(train + valid + test) == list(range(len(deduped)))

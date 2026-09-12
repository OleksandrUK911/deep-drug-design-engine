import pytest
import torch
from torch_geometric.data import Batch

from ml.data.featurizer import NUM_ATOM_FEATURES, smiles_to_graph
from ml.explainability import atom_importance, explain_prediction
from ml.models.gnn import build_model
from ml.uncertainty import mc_dropout_predict


def _model(dropout=0.3):
    torch.manual_seed(0)
    return build_model("gin", in_dim=NUM_ATOM_FEATURES, hidden_dim=16, num_layers=2, dropout=dropout)


def test_mc_dropout_returns_expected_keys_and_shapes():
    model = _model()
    batch = Batch.from_data_list([smiles_to_graph("CCO"), smiles_to_graph("c1ccccc1")])
    out = mc_dropout_predict(model, batch, n_samples=10)
    assert set(out) == {"mean_logit", "std_logit", "mean_prob", "std_prob", "samples"}
    assert out["mean_prob"].shape == (2,)
    assert out["samples"].shape == (10, 2)
    assert (out["std_prob"] >= 0).all()


def test_mc_dropout_zero_dropout_gives_zero_std():
    """With dropout=0, every stochastic pass is identical, so the
    estimated uncertainty must collapse to exactly zero — this is what
    distinguishes MC Dropout uncertainty from just noise in the code."""
    model = _model(dropout=0.0)
    batch = Batch.from_data_list([smiles_to_graph("CCO")])
    out = mc_dropout_predict(model, batch, n_samples=10)
    assert out["std_prob"].item() == pytest.approx(0.0, abs=1e-6)


def test_mc_dropout_nonzero_dropout_gives_nonzero_std():
    model = _model(dropout=0.5)
    batch = Batch.from_data_list([smiles_to_graph("c1ccccc1C(=O)O")])
    out = mc_dropout_predict(model, batch, n_samples=30)
    assert out["std_prob"].item() > 0.0


def test_explain_prediction_returns_per_atom_importance():
    model = _model(dropout=0.0)
    model.eval()
    g = smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O")  # aspirin, 13 heavy atoms
    explanation = explain_prediction(model, g.x, g.edge_index, epochs=20)
    importance = atom_importance(explanation)
    assert importance.shape == (g.x.shape[0],)
    assert (importance >= 0).all()

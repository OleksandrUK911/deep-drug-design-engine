import torch
from torch_geometric.loader import DataLoader

from ml.data.featurizer import NUM_ATOM_FEATURES, smiles_to_graph
from ml.models.gnn import GCNPropertyModel, GINPropertyModel, build_model

SAMPLE_SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "c1ccncc1"]


def _make_batch():
    graphs = [smiles_to_graph(s, y=float(i % 2)) for i, s in enumerate(SAMPLE_SMILES)]
    loader = DataLoader(graphs, batch_size=len(graphs))
    return next(iter(loader))


def test_gcn_forward_shape():
    batch = _make_batch()
    model = GCNPropertyModel(in_dim=NUM_ATOM_FEATURES, hidden_dim=16, num_layers=2, out_dim=1)
    out = model(batch.x, batch.edge_index, batch.batch)
    assert out.shape == (len(SAMPLE_SMILES), 1)


def test_gin_forward_shape():
    batch = _make_batch()
    model = GINPropertyModel(in_dim=NUM_ATOM_FEATURES, hidden_dim=16, num_layers=2, out_dim=1)
    out = model(batch.x, batch.edge_index, batch.batch)
    assert out.shape == (len(SAMPLE_SMILES), 1)


def test_build_model_rejects_unknown_architecture():
    try:
        build_model("not_a_real_architecture", in_dim=NUM_ATOM_FEATURES)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_gcn_multitask_output_dim():
    batch = _make_batch()
    model = GCNPropertyModel(in_dim=NUM_ATOM_FEATURES, hidden_dim=16, num_layers=2, out_dim=12)
    out = model(batch.x, batch.edge_index, batch.batch)
    assert out.shape == (len(SAMPLE_SMILES), 12)


def test_training_step_reduces_loss():
    """Sanity check: a handful of gradient steps on a tiny fixed batch
    should reduce the loss — catches a broken forward/backward wiring
    without needing a full training run."""
    batch = _make_batch()
    model = build_model("gin", in_dim=NUM_ATOM_FEATURES, hidden_dim=16, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]

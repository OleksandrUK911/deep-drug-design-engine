import torch

from ml.data.featurizer import ATOM_SYMBOLS, smiles_to_graph
from ml.training.pretrain_ssl import NUM_ATOM_SYMBOL_CLASSES, SSLPretrainModel, mask_graph


def test_mask_graph_masks_requested_fraction():
    graph = smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O")  # aspirin, 13 atoms
    generator = torch.Generator().manual_seed(0)
    masked, mask_idx, target = mask_graph(graph, mask_fraction=0.5, generator=generator)

    assert mask_idx.numel() == int(13 * 0.5)
    assert target.shape == mask_idx.shape
    assert (target >= 0).all() and (target < NUM_ATOM_SYMBOL_CLASSES).all()
    # masked positions are zeroed out in the returned graph
    assert torch.all(masked.x[mask_idx] == 0.0)
    # unmasked positions are untouched
    unmasked = torch.tensor([i for i in range(13) if i not in mask_idx.tolist()])
    assert torch.equal(masked.x[unmasked], graph.x[unmasked])


def test_mask_graph_target_matches_original_symbol():
    graph = smiles_to_graph("CCO")  # C, C, O
    generator = torch.Generator().manual_seed(1)
    _, mask_idx, target = mask_graph(graph, mask_fraction=1.0, generator=generator)
    # every atom's target should equal the argmax of its original symbol one-hot
    expected = graph.x[mask_idx, : len(ATOM_SYMBOLS) + 1].argmax(dim=-1)
    assert torch.equal(target, expected)


def test_ssl_pretrain_model_forward_shape():
    graph = smiles_to_graph("c1ccccc1")
    model = SSLPretrainModel(in_dim=graph.x.shape[1], hidden_dim=16, num_layers=2)
    logits = model(graph.x, graph.edge_index)
    assert logits.shape == (graph.x.shape[0], NUM_ATOM_SYMBOL_CLASSES)


def test_pretrained_encoder_state_dict_loads_into_property_model():
    from ml.models.gnn import GINPropertyModel

    graph = smiles_to_graph("CCO")
    pretrain_model = SSLPretrainModel(in_dim=graph.x.shape[1], hidden_dim=16, num_layers=2)
    property_model = GINPropertyModel(in_dim=graph.x.shape[1], hidden_dim=16, num_layers=2)

    # Must not raise — this is the exact transfer path train_property_model.py uses.
    property_model.load_pretrained_encoder(pretrain_model.encoder.state_dict())

    out = property_model(graph.x, graph.edge_index, torch.zeros(graph.x.shape[0], dtype=torch.long))
    assert out.shape == (1, 1)

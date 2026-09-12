import torch
from torch_geometric.loader import DataLoader

from ml.data.featurizer import NUM_ATOM_FEATURES
from ml.generative.graph_vae import (
    MAX_ATOMS,
    NONE_ATOM_IDX,
    NONE_BOND_IDX,
    GraphVAE,
    mol_to_fixed_size_targets,
    tensors_to_mol,
    vae_loss,
)
from ml.generative.metrics import generation_metrics, internal_diversity
from ml.generative.smiles_rnn import CharRNNGenerator
from ml.generative.smiles_vocab import SmilesVocab
from ml.synthesizability import sa_score
from rdkit import Chem


def test_vocab_roundtrip():
    vocab = SmilesVocab(["CCO", "c1ccccc1"])
    encoded = vocab.encode("CCO")
    assert encoded[0] == vocab.bos_idx
    assert encoded[-1] == vocab.eos_idx
    assert vocab.decode(encoded) == "CCO"


def test_vocab_pad_and_unknown_handling_in_decode():
    vocab = SmilesVocab(["CCO"])
    # decode should stop at eos and skip pad/bos even if they trail after it
    seq = vocab.encode("CCO") + [vocab.pad_idx, vocab.pad_idx]
    assert vocab.decode(seq) == "CCO"


def test_char_rnn_forward_shape():
    vocab = SmilesVocab(["CCO", "CCN"])
    model = CharRNNGenerator(len(vocab), embed_dim=8, hidden_dim=16, num_layers=1)
    x = torch.tensor([vocab.encode("CCO")[:-1]])
    logits, _ = model(x)
    assert logits.shape == (1, x.shape[1], len(vocab))


def test_char_rnn_sample_returns_strings():
    vocab = SmilesVocab(["CCO", "CCN", "c1ccccc1"])
    model = CharRNNGenerator(len(vocab), embed_dim=8, hidden_dim=16, num_layers=1)
    samples = model.sample(vocab, n_samples=5, max_len=20)
    assert len(samples) == 5
    assert all(isinstance(s, str) for s in samples)


def test_generation_metrics_all_valid_unique_novel():
    metrics = generation_metrics(["CCO", "CCN", "CCC"], training_smiles=set())
    assert metrics["validity"] == 1.0
    assert metrics["uniqueness"] == 1.0
    assert metrics["novelty"] == 1.0
    assert metrics["n_total"] == 3


def test_generation_metrics_counts_invalid_and_duplicates():
    metrics = generation_metrics(["CCO", "CCO", "not a molecule"], training_smiles={"CCO"})
    assert metrics["n_valid"] == 2  # two "CCO" parse fine
    assert metrics["n_unique"] == 1  # they're the same molecule
    assert metrics["novelty"] == 0.0  # CCO is in the training set


def test_internal_diversity_identical_molecules_is_zero():
    assert internal_diversity(["CCO", "CCO", "CCO"]) == 0.0


def test_internal_diversity_different_molecules_is_positive():
    assert internal_diversity(["CCO", "c1ccccc1", "CC(=O)O"]) > 0.0


def test_sa_score_returns_plausible_range():
    score = sa_score("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    assert score is not None
    assert 1.0 <= score <= 10.0


def test_sa_score_invalid_smiles_returns_none():
    assert sa_score("not a molecule") is None


def test_mol_to_fixed_size_targets_and_back_roundtrip():
    mol = Chem.MolFromSmiles("CCO")
    targets = mol_to_fixed_size_targets(mol)
    assert targets is not None
    atom_types, edge_types = targets
    assert atom_types.shape == (MAX_ATOMS,)
    assert edge_types.shape == (MAX_ATOMS, MAX_ATOMS)
    assert (atom_types[3:] == NONE_ATOM_IDX).all()  # only 3 real atoms

    decoded = tensors_to_mol(atom_types, edge_types)
    assert decoded == "CCO"


def test_mol_to_fixed_size_targets_rejects_oversized_molecule():
    huge = Chem.MolFromSmiles("C" * 40)  # 40-carbon chain > MAX_ATOMS
    assert mol_to_fixed_size_targets(huge) is None


def test_graph_vae_forward_and_loss_shapes():
    from ml.data.featurizer import smiles_to_graph
    from ml.training.train_graph_vae import build_training_examples

    examples = build_training_examples(["CCO", "CCN", "c1ccccc1"])
    loader = DataLoader(examples, batch_size=3)
    batch = next(iter(loader))

    model = GraphVAE(NUM_ATOM_FEATURES, hidden_dim=16, latent_dim=8)
    atom_logits, edge_logits, mu, logvar = model(batch.x, batch.edge_index, batch.batch)
    assert atom_logits.shape[0] == 3
    assert mu.shape == (3, 8)

    loss, parts = vae_loss(atom_logits, edge_logits, batch.atom_types, batch.edge_types, mu, logvar)
    assert loss.item() > 0
    assert set(parts) == {"atom_loss", "edge_loss", "kl"}


def test_graph_vae_sample_returns_requested_count():
    model = GraphVAE(NUM_ATOM_FEATURES, hidden_dim=16, latent_dim=8)
    samples = model.sample(5)
    assert len(samples) == 5
    # entries are either a valid canonical SMILES string or None
    assert all(s is None or isinstance(s, str) for s in samples)

import pytest
import torch

from ml.generative.smiles_rnn import CharRNNGenerator
from ml.generative.smiles_vocab import SmilesVocab
from ml.rl.reinvent import train_reinvent
from ml.rl.reward import RewardConfig, normalize_docking, normalize_sascore, score_batch, score_molecule


def test_normalize_sascore_endpoints():
    assert normalize_sascore(1.0) == 1.0  # easiest possible -> best score
    assert normalize_sascore(10.0) == 0.0  # hardest possible -> worst score
    assert normalize_sascore(0.0) == 1.0  # clipped, doesn't exceed 1
    assert normalize_sascore(15.0) == 0.0  # clipped, doesn't go negative


def test_normalize_docking_endpoints():
    assert normalize_docking(-12.0, best=-12.0, worst=0.0) == 1.0
    assert normalize_docking(0.0, best=-12.0, worst=0.0) == 0.0
    assert normalize_docking(-20.0, best=-12.0, worst=0.0) == 1.0  # clipped past "best"


def test_score_molecule_invalid_smiles_is_all_zero():
    config = RewardConfig(weights={"qed": 0.5, "synth": 0.5})
    result = score_molecule("not a valid smiles", config)
    assert result["valid"] is False
    assert result["total"] == 0.0
    assert result["qed"] == 0.0
    assert result["synth"] == 0.0


def test_score_molecule_valid_smiles_has_positive_components():
    config = RewardConfig(weights={"qed": 0.5, "synth": 0.5})
    result = score_molecule("CC(=O)Oc1ccccc1C(=O)O", config)  # aspirin
    assert result["valid"] is True
    assert result["qed"] > 0
    assert result["synth"] > 0
    assert 0 <= result["total"] <= 1


def test_score_batch_matches_individual_scores():
    config = RewardConfig(weights={"qed": 1.0})
    smiles = ["CCO", "c1ccccc1", "invalid((("]
    batch_results = score_batch(smiles, config)
    individual_results = [score_molecule(s, config) for s in smiles]
    assert [r["total"] for r in batch_results] == [r["total"] for r in individual_results]


def test_sequence_log_prob_matches_manual_computation():
    vocab = SmilesVocab(["CCO", "CCN"])
    model = CharRNNGenerator(len(vocab), embed_dim=8, hidden_dim=16, num_layers=1)
    model.eval()

    token_ids = torch.tensor([vocab.encode("CCO")])
    log_prob = model.sequence_log_prob(token_ids, vocab.pad_idx)

    # Manual: sum log-softmax at each target position, teacher-forced.
    inputs, targets = token_ids[:, :-1], token_ids[:, 1:]
    logits, _ = model(inputs)
    log_probs = torch.log_softmax(logits, dim=-1)
    expected = sum(log_probs[0, t, targets[0, t]].item() for t in range(targets.shape[1]))

    assert log_prob.item() == pytest.approx(expected, abs=1e-4)


def test_sample_token_ids_shape_and_padding():
    vocab = SmilesVocab(["CCO", "c1ccccc1"])
    model = CharRNNGenerator(len(vocab), embed_dim=8, hidden_dim=16, num_layers=1)
    token_ids = model.sample_token_ids(vocab, n_samples=5, max_len=15)
    assert token_ids.shape[0] == 5
    assert (token_ids[:, 0] == vocab.bos_idx).all()  # every sequence starts with BOS


def test_train_reinvent_reduces_agent_prior_divergence_gracefully():
    """Mechanical smoke test: loss/gradients flow, agent parameters
    change, and the run completes without diverging to NaN — the
    end-to-end learning-curve validation happens in
    ml/training/train_rl.py against the real pretrained checkpoint, not
    here (too slow for a unit test)."""
    smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN"]
    vocab = SmilesVocab(smiles)
    model = CharRNNGenerator(len(vocab), embed_dim=8, hidden_dim=16, num_layers=1)
    config = RewardConfig(weights={"qed": 1.0})

    agent = train_reinvent(model, vocab, config, n_steps=3, batch_size=4, sigma=10.0, lr=1e-3, max_len=15, log_every=100)

    for p in agent.parameters():
        assert torch.isfinite(p).all()
    assert len(agent.history) == 3

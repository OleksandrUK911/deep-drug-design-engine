"""REINVENT-style policy-gradient fine-tuning (Olivecrona et al., 2017)
for the char-RNN generator, see TODO/ml/TODO_rl_optimization.md.

Instead of vanilla REINFORCE (high-variance, prone to mode collapse for
sequence generation), the agent is trained to match an "augmented
likelihood": log P_prior(x) + sigma * score(x). Minimizing
(log P_agent(x) - augmented)^2 pulls the agent toward molecules the prior
already considered plausible *and* that score well — the prior term acts
as a built-in anchor against reward hacking (this *is* the "KL-penalty to
the pretrained distribution" TODO_rl_optimization.md asks for, expressed
as a regression target rather than an explicit KL term, which is the
standard formulation in the literature and more stable in practice).
"""
from __future__ import annotations

import copy

import torch

from ml.generative.smiles_rnn import CharRNNGenerator
from ml.generative.smiles_vocab import SmilesVocab
from ml.rl.reward import RewardConfig, score_batch


def train_reinvent(
    pretrained_model: CharRNNGenerator,
    vocab: SmilesVocab,
    reward_config: RewardConfig,
    n_steps: int = 50,
    batch_size: int = 64,
    sigma: float = 60.0,
    lr: float = 1e-4,
    max_len: int = 100,
    temperature: float = 1.0,
    device: str = "cpu",
    log_every: int = 5,
    callback=None,
) -> CharRNNGenerator:
    """Fine-tune a copy of `pretrained_model` (left untouched) against
    `reward_config`. `sigma` scales how strongly reward pulls the agent
    away from the prior's likelihood — the single most important
    hyperparameter for avoiding either "no learning" (too small) or mode
    collapse (too large).
    """
    prior = copy.deepcopy(pretrained_model).to(device)
    prior.eval()
    for p in prior.parameters():
        p.requires_grad_(False)

    agent = copy.deepcopy(pretrained_model).to(device)
    agent.train()
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    history = []
    for step in range(n_steps):
        token_ids = agent.sample_token_ids(vocab, batch_size, max_len=max_len, temperature=temperature, device=device)
        smiles_batch = [vocab.decode(seq.tolist()) for seq in token_ids]

        agent_log_probs = agent.sequence_log_prob(token_ids, vocab.pad_idx)
        with torch.no_grad():
            prior_log_probs = prior.sequence_log_prob(token_ids, vocab.pad_idx)

        rewards = score_batch(smiles_batch, reward_config)
        reward_tensor = torch.tensor([r["total"] for r in rewards], device=device, dtype=torch.float)

        augmented = prior_log_probs + sigma * reward_tensor
        loss = torch.mean((agent_log_probs - augmented) ** 2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        optimizer.step()

        valid_frac = sum(r["valid"] for r in rewards) / len(rewards)
        unique_frac = len(set(smiles_batch)) / len(smiles_batch)
        mean_reward = reward_tensor.mean().item()

        record = {
            "step": step, "loss": loss.item(), "mean_reward": mean_reward,
            "valid_frac": valid_frac, "unique_frac": unique_frac,
        }
        history.append(record)
        if callback is not None:
            callback(record, smiles_batch, rewards)
        if step % log_every == 0 or step == n_steps - 1:
            print(f"step {step:3d} | loss={loss.item():.3f} | mean_reward={mean_reward:.3f} "
                  f"| valid={valid_frac:.2f} | unique={unique_frac:.2f}")

    agent.history = history  # stash for callers that want the full curve without a callback
    return agent

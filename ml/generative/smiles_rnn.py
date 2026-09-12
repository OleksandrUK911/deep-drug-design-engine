"""Char-level LSTM SMILES generator — the "fast path to a working
end-to-end demo" baseline described in TODO/ml/TODO_generative_model.md,
trained with standard teacher-forced next-character prediction and sampled
autoregressively with temperature.
"""
from __future__ import annotations

import torch
from torch import nn

from ml.generative.smiles_vocab import SmilesVocab


class CharRNNGenerator(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.head(out)
        return logits, hidden

    @torch.no_grad()
    def sample(self, vocab: SmilesVocab, n_samples: int, max_len: int = 100, temperature: float = 1.0, device: str = "cpu") -> list[str]:
        self.eval()
        tokens = torch.full((n_samples, 1), vocab.bos_idx, dtype=torch.long, device=device)
        hidden = None
        finished = torch.zeros(n_samples, dtype=torch.bool, device=device)
        generated = [[] for _ in range(n_samples)]

        current = tokens
        for _ in range(max_len):
            logits, hidden = self.forward(current, hidden)
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            for i in range(n_samples):
                if not finished[i]:
                    generated[i].append(next_tokens[i].item())
                    if next_tokens[i].item() == vocab.eos_idx:
                        finished[i] = True

            if finished.all():
                break
            current = next_tokens.unsqueeze(-1)

        return [vocab.decode(seq) for seq in generated]

    @torch.no_grad()
    def sample_token_ids(self, vocab: SmilesVocab, n_samples: int, max_len: int = 100, temperature: float = 1.0, device: str = "cpu") -> torch.Tensor:
        """Like `sample`, but returns padded token-id sequences (including
        the trailing EOS, padded with pad_idx after it) instead of decoded
        strings — the form ml.rl.reinvent needs to recompute log-likelihood
        via teacher forcing on exactly what was sampled.
        """
        self.eval()
        current = torch.full((n_samples, 1), vocab.bos_idx, dtype=torch.long, device=device)
        hidden = None
        finished = torch.zeros(n_samples, dtype=torch.bool, device=device)
        all_tokens = [current]

        for _ in range(max_len):
            logits, hidden = self.forward(current, hidden)
            probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)  # [n_samples, 1]
            next_tokens[finished] = vocab.pad_idx
            finished = finished | (next_tokens.squeeze(-1) == vocab.eos_idx)
            all_tokens.append(next_tokens)
            current = next_tokens
            if finished.all():
                break

        return torch.cat(all_tokens, dim=1)  # [n_samples, seq_len] including leading BOS

    def sequence_log_prob(self, token_ids: torch.Tensor, pad_idx: int) -> torch.Tensor:
        """Teacher-forced total log-likelihood of each sequence in
        `token_ids` (shape [batch, seq_len], BOS-first) under this model's
        *current* parameters. Differentiable — this is what both the agent
        (with grad) and the frozen prior (under torch.no_grad, see
        ml/rl/reinvent.py) use to score the same sampled sequences.
        """
        inputs, targets = token_ids[:, :-1], token_ids[:, 1:]
        logits, _ = self.forward(inputs)
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        mask = (targets != pad_idx).float()
        return (token_log_probs * mask).sum(dim=1)

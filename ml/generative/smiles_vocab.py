"""Character-level vocabulary for the SMILES generator (see
TODO/ml/TODO_generative_model.md). Built from whatever training corpus is
passed in, not hardcoded, so it stays correct if the training set changes.
"""
from __future__ import annotations

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
SPECIAL_TOKENS = [PAD, BOS, EOS]


class SmilesVocab:
    def __init__(self, smiles_list: list[str]):
        chars = sorted({c for s in smiles_list for c in s})
        self.itos = SPECIAL_TOKENS + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_idx = self.stoi[PAD]
        self.bos_idx = self.stoi[BOS]
        self.eos_idx = self.stoi[EOS]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, smiles: str) -> list[int]:
        return [self.bos_idx] + [self.stoi[c] for c in smiles] + [self.eos_idx]

    def decode(self, indices: list[int]) -> str:
        chars = []
        for i in indices:
            if i == self.eos_idx:
                break
            if i in (self.bos_idx, self.pad_idx):
                continue
            chars.append(self.itos[i])
        return "".join(chars)

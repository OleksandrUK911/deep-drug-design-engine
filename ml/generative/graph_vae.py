"""Fixed-size Graph VAE for de novo molecule generation (see
TODO/ml/TODO_generative_model.md).

Simplification, stated explicitly: unlike the original GraphVAE (Simonovsky
& Komodakis, 2018), this implementation does **not** do permutation-invariant
graph matching for the reconstruction loss — it reconstructs atoms/bonds in
whatever order RDKit assigns when parsing the canonical SMILES. That makes
training tractable on CPU without an approximate-graph-matching solver, at
the cost of penalizing structurally-correct-but-differently-ordered
reconstructions during training. It is a known, deliberate baseline
simplification, not an oversight — full graph matching is a natural next
step, not a blocker for having a working generator.

Molecules are capped at `MAX_ATOMS` heavy atoms and restricted to the
element set in `ml.data.featurizer.ATOM_SYMBOLS` (covers the overwhelming
majority of drug-like molecules) so the decoder's output size is fixed.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import RWMol
from torch import nn
from torch_geometric.nn import global_mean_pool

from ml.data.featurizer import ATOM_SYMBOLS, BOND_TYPES, canonical_smiles
from ml.models.encoder import GINEncoder

MAX_ATOMS = 25
NUM_ATOM_TYPES = len(ATOM_SYMBOLS) + 1  # + "no atom" (padding slot)
NUM_BOND_TYPES = len(BOND_TYPES) + 1  # + "no bond"
NONE_ATOM_IDX = len(ATOM_SYMBOLS)
NONE_BOND_IDX = len(BOND_TYPES)

_SYMBOL_TO_ATOMIC_NUM = {
    "C": 6, "N": 7, "O": 8, "S": 16, "F": 9, "Cl": 17,
    "Br": 35, "I": 53, "P": 15, "B": 5, "Si": 14,
}
_RDKIT_BOND_TYPE = {0: Chem.BondType.SINGLE, 1: Chem.BondType.DOUBLE, 2: Chem.BondType.TRIPLE, 3: Chem.BondType.AROMATIC}


def mol_to_fixed_size_targets(mol) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """RDKit Mol -> (atom_types [MAX_ATOMS], edge_types [MAX_ATOMS, MAX_ATOMS])
    in RDKit's own atom order. Returns None if the molecule is too big or
    contains an element outside ATOM_SYMBOLS."""
    n = mol.GetNumAtoms()
    if n > MAX_ATOMS:
        return None

    atom_types = torch.full((MAX_ATOMS,), NONE_ATOM_IDX, dtype=torch.long)
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() not in ATOM_SYMBOLS:
            return None
        atom_types[i] = ATOM_SYMBOLS.index(atom.GetSymbol())

    edge_types = torch.full((MAX_ATOMS, MAX_ATOMS), NONE_BOND_IDX, dtype=torch.long)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_idx = BOND_TYPES.index(bond.GetBondType()) if bond.GetBondType() in BOND_TYPES else None
        if bond_idx is None:
            return None
        edge_types[i, j] = bond_idx
        edge_types[j, i] = bond_idx

    return atom_types, edge_types


def tensors_to_mol(atom_types: torch.Tensor, edge_types: torch.Tensor) -> Optional[str]:
    """Decode argmax atom/edge type tensors into a SMILES string, or None
    if the resulting molecule fails RDKit sanitization (an expected, not
    exceptional, outcome for a simple non-autoregressive decoder)."""
    real_atoms = [i for i in range(atom_types.size(0)) if atom_types[i].item() != NONE_ATOM_IDX]
    if len(real_atoms) < 1:
        return None

    rw = RWMol()
    idx_map = {}
    for i in real_atoms:
        symbol = ATOM_SYMBOLS[atom_types[i].item()]
        idx_map[i] = rw.AddAtom(Chem.Atom(_SYMBOL_TO_ATOMIC_NUM[symbol]))

    for a, i in enumerate(real_atoms):
        for j in real_atoms[a + 1 :]:
            bond_idx = edge_types[i, j].item()
            if bond_idx != NONE_BOND_IDX:
                try:
                    rw.AddBond(idx_map[i], idx_map[j], _RDKIT_BOND_TYPE[bond_idx])
                except Exception:
                    return None

    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return canonical_smiles(Chem.MolToSmiles(mol))


class GraphVAE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, latent_dim: int = 56, num_encoder_layers: int = 3):
        super().__init__()
        self.encoder = GINEncoder(in_dim, hidden_dim, num_encoder_layers)
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

        decoder_hidden = 512
        self.decoder_trunk = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.ReLU(),
        )
        self.atom_head = nn.Linear(decoder_hidden, MAX_ATOMS * NUM_ATOM_TYPES)
        self.edge_head = nn.Linear(decoder_hidden, MAX_ATOMS * MAX_ATOMS * NUM_BOND_TYPES)

    def encode(self, x, edge_index, batch):
        node_emb = self.encoder(x, edge_index)
        graph_emb = global_mean_pool(node_emb, batch)
        return self.to_mu(graph_emb), self.to_logvar(graph_emb)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = self.decoder_trunk(z)
        atom_logits = self.atom_head(h).view(-1, MAX_ATOMS, NUM_ATOM_TYPES)
        edge_logits = self.edge_head(h).view(-1, MAX_ATOMS, MAX_ATOMS, NUM_BOND_TYPES)
        # Symmetrize edge logits so the decoder can't just memorize a
        # direction-dependent bond pattern for an undirected graph.
        edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2
        return atom_logits, edge_logits

    def forward(self, x, edge_index, batch):
        mu, logvar = self.encode(x, edge_index, batch)
        z = self.reparameterize(mu, logvar)
        atom_logits, edge_logits = self.decode(z)
        return atom_logits, edge_logits, mu, logvar

    @torch.no_grad()
    def sample(self, n_samples: int, device: str = "cpu") -> list[Optional[str]]:
        self.eval()
        latent_dim = self.to_mu.out_features
        z = torch.randn(n_samples, latent_dim, device=device)
        atom_logits, edge_logits = self.decode(z)
        atom_types = atom_logits.argmax(dim=-1)
        edge_types = edge_logits.argmax(dim=-1)
        return [tensors_to_mol(atom_types[i], edge_types[i]) for i in range(n_samples)]


def vae_loss(atom_logits, edge_logits, atom_targets, edge_targets, mu, logvar, kl_weight: float = 1.0):
    atom_loss = F.cross_entropy(atom_logits.reshape(-1, NUM_ATOM_TYPES), atom_targets.reshape(-1))

    # Only the upper triangle carries independent information (edges are
    # symmetric); scoring the full matrix would double-count every bond.
    n = edge_logits.size(1)
    triu_i, triu_j = torch.triu_indices(n, n, offset=1)
    edge_logits_triu = edge_logits[:, triu_i, triu_j, :]
    edge_targets_triu = edge_targets[:, triu_i, triu_j]
    edge_loss = F.cross_entropy(edge_logits_triu.reshape(-1, NUM_BOND_TYPES), edge_targets_triu.reshape(-1))

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = atom_loss + edge_loss + kl_weight * kl
    return total, {"atom_loss": atom_loss.item(), "edge_loss": edge_loss.item(), "kl": kl.item()}

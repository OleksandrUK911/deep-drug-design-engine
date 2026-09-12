"""PyTorch Geometric Dataset for the MoleculeNet-style property CSVs
(Tox21, BBBP, ESOL). See TODO/data/TODO_preprocessing_pipeline.md and
TODO/data/TODO_quality_validation.md.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
from torch_geometric.data import InMemoryDataset

from .featurizer import smiles_to_graph
from .splits import dedup_smiles


def _read_csv(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return pd.read_csv(f)
    return pd.read_csv(path)


class MoleculePropertyDataset(InMemoryDataset):
    """Loads a MoleculeNet-style CSV with a `smiles` column and one or more
    label columns, drops invalid/duplicate SMILES, and featurizes the rest
    into PyG graphs.

    Rows with a missing label (NaN) for the requested target are dropped —
    Tox21 in particular has many missing labels per task.
    """

    def __init__(self, csv_path: str | Path, smiles_col: str, label_cols: list[str]):
        self.csv_path = Path(csv_path)
        self.smiles_col = smiles_col
        self.label_cols = label_cols
        super().__init__(root=None)
        self.data, self.slices = self._process()

    def _process(self):
        df = _read_csv(self.csv_path)
        df = df.dropna(subset=[self.smiles_col] + self.label_cols, how="any" if len(self.label_cols) == 1 else "all")

        smiles = df[self.smiles_col].tolist()
        keep_idx = dedup_smiles(smiles)
        df = df.iloc[keep_idx].reset_index(drop=True)

        data_list = []
        for _, row in df.iterrows():
            label = float(row[self.label_cols[0]]) if len(self.label_cols) == 1 else None
            graph = smiles_to_graph(row[self.smiles_col], y=label)
            if graph is None:
                continue
            if len(self.label_cols) > 1:
                import torch

                graph.y = torch.tensor(
                    [row[c] if pd.notna(row[c]) else float("nan") for c in self.label_cols],
                    dtype=torch.float,
                ).unsqueeze(0)
            data_list.append(graph)

        return self.collate(data_list)

    @property
    def smiles(self) -> list[str]:
        """Not stored on the collated Data batch — re-derive by re-reading
        the source CSV is the caller's job if needed for scaffold splitting
        before featurization; this property is intentionally omitted from
        the fast path to avoid keeping two copies of the string list."""
        raise NotImplementedError(
            "Use featurize_with_smiles() if you need SMILES alongside graphs "
            "(e.g. for scaffold_split before training)."
        )


def load_smiles_and_labels(
    csv_path: str | Path, smiles_col: str, label_cols: list[str]
) -> tuple[list[str], "pd.DataFrame"]:
    """Read + dedup a property CSV, returning the deduplicated SMILES list
    and the corresponding label rows — the shape `scaffold_split` and
    `MoleculePropertyDataset` both expect as input."""
    df = _read_csv(Path(csv_path))
    df = df.dropna(subset=[smiles_col])
    smiles = df[smiles_col].tolist()
    keep_idx = dedup_smiles(smiles)
    df = df.iloc[keep_idx].reset_index(drop=True)
    return df[smiles_col].tolist(), df[label_cols]

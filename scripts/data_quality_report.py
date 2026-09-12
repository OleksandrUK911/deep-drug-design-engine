#!/usr/bin/env python3
"""Print a basic data-quality report for a SMILES CSV: validity rate,
duplicate rate, molecule size distribution, and element composition.
See TODO/data/TODO_quality_validation.md.

Usage:
    python scripts/data_quality_report.py data/raw/BBBP.csv --smiles-col smiles
"""
from __future__ import annotations

import argparse
import gzip
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from rdkit import Chem

from ml.data.featurizer import canonical_smiles


def read_csv(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return pd.read_csv(f)
    return pd.read_csv(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--smiles-col", default="smiles")
    args = parser.parse_args()

    df = read_csv(args.csv_path)
    total = len(df)
    smiles_list = df[args.smiles_col].dropna().tolist()

    valid_canon: list[str] = []
    invalid_count = 0
    atom_counts: list[int] = []
    element_counter: Counter[str] = Counter()

    for smiles in smiles_list:
        canon = canonical_smiles(smiles)
        if canon is None:
            invalid_count += 1
            continue
        valid_canon.append(canon)
        mol = Chem.MolFromSmiles(canon)
        atom_counts.append(mol.GetNumAtoms())
        element_counter.update(atom.GetSymbol() for atom in mol.GetAtoms())

    unique_count = len(set(valid_canon))
    duplicate_count = len(valid_canon) - unique_count

    print(f"Dataset: {args.csv_path}")
    print(f"  Total rows:            {total}")
    print(f"  Missing SMILES:        {total - len(smiles_list)}")
    print(f"  Invalid SMILES:        {invalid_count} ({invalid_count / max(len(smiles_list), 1):.1%})")
    print(f"  Valid SMILES:          {len(valid_canon)}")
    print(f"  Duplicates (valid):    {duplicate_count} ({duplicate_count / max(len(valid_canon), 1):.1%})")
    print(f"  Unique valid SMILES:   {unique_count}")
    if atom_counts:
        print(f"  Atoms per molecule:    min={min(atom_counts)} max={max(atom_counts)} "
              f"mean={sum(atom_counts) / len(atom_counts):.1f}")
    top_elements = element_counter.most_common(10)
    print(f"  Top elements:          {', '.join(f'{el}={n}' for el, n in top_elements)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

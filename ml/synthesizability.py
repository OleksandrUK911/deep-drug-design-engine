"""Synthetic Accessibility Score (SAScore) via RDKit's Contrib module (see
TODO/ml/TODO_synthesizability_filtering.md). SAScore ranges 1 (easy to
synthesize) to 10 (very hard) — this exists specifically to catch the
common de-novo-generation failure mode of proposing chemically valid but
practically unsynthesizable molecules.

RDKit ships sascorer.py under Contrib/ rather than as an importable
top-level module, so we import it by file path instead of requiring a
separate pip package.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Optional

import rdkit
from rdkit import Chem


def _load_sascorer():
    contrib_path = Path(rdkit.__file__).parent / "Contrib" / "SA_Score" / "sascorer.py"
    if not contrib_path.exists():
        raise ImportError(
            f"RDKit SA_Score contrib script not found at {contrib_path}. "
            "Reinstall rdkit or point this at your RDKit Contrib directory."
        )
    spec = importlib.util.spec_from_file_location("sascorer", contrib_path)
    module = importlib.util.module_from_spec(spec)
    # sascorer.py loads fpscores.pkl.gz relative to its own __file__ via os.path.dirname,
    # so it must be executed with __file__ set correctly, which spec_from_file_location does.
    spec.loader.exec_module(module)
    return module


_sascorer = None


def sa_score(smiles: str) -> Optional[float]:
    """Return the SAScore for a SMILES string, or None if it fails to
    parse. Lazily loads the scoring model (reads a ~1MB pickle) once per
    process."""
    global _sascorer
    if _sascorer is None:
        _sascorer = _load_sascorer()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return float(_sascorer.calculateScore(mol))

"""Generation quality metrics (MOSES/GuacaMol-style, see
TODO/ml/TODO_evaluation_metrics.md): validity, uniqueness, novelty.
"""
from __future__ import annotations

from rdkit import Chem

from ml.data.featurizer import canonical_smiles


def generation_metrics(generated_smiles: list[str], training_smiles: set[str]) -> dict:
    """Standard generative-chemistry benchmark trio:
    - validity: fraction RDKit can parse+sanitize
    - uniqueness: fraction of the *valid* molecules that are distinct
    - novelty: fraction of the *unique valid* molecules not seen in training
    """
    n_total = len(generated_smiles)
    valid_canon = [canonical_smiles(s) for s in generated_smiles]
    valid_canon = [s for s in valid_canon if s is not None]
    n_valid = len(valid_canon)

    unique_canon = set(valid_canon)
    n_unique = len(unique_canon)

    n_novel = len(unique_canon - training_smiles)

    return {
        "validity": n_valid / n_total if n_total else 0.0,
        "uniqueness": n_unique / n_valid if n_valid else 0.0,
        "novelty": n_novel / n_unique if n_unique else 0.0,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_unique": n_unique,
        "n_novel": n_novel,
    }


def internal_diversity(smiles_list: list[str]) -> float:
    """Mean pairwise Tanimoto distance (1 - similarity) over Morgan
    fingerprints — a cheap stand-in for MOSES's internal diversity metric
    that avoids pulling in the full MOSES/FCD dependency stack."""
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols = [m for m in mols if m is not None]
    if len(mols) < 2:
        return 0.0

    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024) for m in mols]
    total, count = 0.0, 0
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        total += sum(1 - s for s in sims)
        count += len(sims)
    return total / count if count else 0.0

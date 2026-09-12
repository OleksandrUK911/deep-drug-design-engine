"""Python wrapper around the AutoDock Vina executable (see
TODO/ml/TODO_docking_scoring.md). Uses the standalone `vina` binary via
subprocess rather than the `vina` PyPI package, because that package
requires compiling against Boost and has no prebuilt wheel for this
platform/Python version — the standalone executable is the officially
distributed alternative and gives identical scoring.

Ligand/receptor PDBQT preparation goes through `meeko`, the actively
maintained successor to the old AutoDockTools scripts.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTMolecule, PDBQTWriterLegacy, RDKitMolCreate

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BIN_DIR = REPO_ROOT / "bin"
VINA_EXE = _BIN_DIR / "vina.exe" if (_BIN_DIR / "vina.exe").exists() else _BIN_DIR / "vina"
CACHE_DIR = REPO_ROOT / "data" / "docking_cache"


def prepare_ligand_pdbqt(smiles: str, out_path: Path, seed: int = 0) -> Path:
    """SMILES -> 3D conformer (ETKDG + MMFF) -> PDBQT, ready for Vina."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError(f"3D embedding failed for {smiles!r}")
    AllChem.MMFFOptimizeMolecule(mol)

    preparator = MoleculePreparation()
    setups = preparator.prepare(mol)
    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setups[0])
    if not is_ok:
        raise ValueError(f"Ligand PDBQT preparation failed for {smiles!r}: {error_msg}")

    out_path.write_text(pdbqt_string)
    return out_path


def prepare_receptor_pdbqt(receptor_pdb: Path, out_basename: Path) -> Path:
    """Protein-only PDB -> PDBQT via meeko's mk_prepare_receptor CLI."""
    cmd = [
        "python", "-m", "meeko.cli.mk_prepare_receptor",
        "--read_pdb", str(receptor_pdb),
        "-o", str(out_basename),
        "-p",  # write PDBQT
        "-a",  # allow bad/missing atoms rather than aborting on the first residue issue
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    pdbqt_path = out_basename.with_suffix(".pdbqt")
    if result.returncode != 0 or not pdbqt_path.exists():
        raise RuntimeError(f"Receptor preparation failed:\nstdout={result.stdout}\nstderr={result.stderr}")
    return pdbqt_path


def _cache_key(receptor_pdbqt: Path, smiles: str, center, box_size, exhaustiveness: int) -> str:
    receptor_hash = hashlib.sha256(receptor_pdbqt.read_bytes()).hexdigest()[:16]
    key_str = f"{receptor_hash}:{smiles}:{center}:{box_size}:{exhaustiveness}"
    return hashlib.sha256(key_str.encode()).hexdigest()[:24]


def dock(
    smiles: str,
    receptor_pdbqt: Path,
    center: tuple[float, float, float],
    box_size: tuple[float, float, float] = (20.0, 20.0, 20.0),
    exhaustiveness: int = 8,
    seed: int = 0,
    use_cache: bool = True,
    work_dir: Optional[Path] = None,
) -> dict:
    """Dock one molecule against one prepared receptor. Returns the best
    (most negative = most favorable) binding-affinity estimate in
    kcal/mol, plus the docked pose as an RDKit Mol with proper bond
    orders (via meeko's RDKitMolCreate).

    Results are cached by (receptor, smiles, box, exhaustiveness) hash —
    docking is by far the most expensive step in the pipeline, and
    re-running it for the same molecule/target/settings is pure waste.
    """
    if not VINA_EXE.exists():
        raise FileNotFoundError(f"Vina executable not found at {VINA_EXE}")

    cache_key = _cache_key(receptor_pdbqt, smiles, center, box_size, exhaustiveness)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if use_cache and cache_file.exists():
        import json

        cached = json.loads(cache_file.read_text())
        cached["from_cache"] = True
        return cached

    work_dir = work_dir or (CACHE_DIR / cache_key)
    work_dir.mkdir(parents=True, exist_ok=True)

    ligand_pdbqt = prepare_ligand_pdbqt(smiles, work_dir / "ligand.pdbqt", seed=seed)
    output_pdbqt = work_dir / "docked.pdbqt"

    cmd = [
        str(VINA_EXE),
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
        "--size_x", str(box_size[0]), "--size_y", str(box_size[1]), "--size_z", str(box_size[2]),
        "--exhaustiveness", str(exhaustiveness),
        "--seed", str(seed),
        "--out", str(output_pdbqt),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"Vina docking failed for {smiles!r}:\n{result.stdout}\n{result.stderr}")

    best_affinity = _parse_best_affinity(result.stdout)

    docked_mol = None
    if output_pdbqt.exists():
        pdbqt_mol = PDBQTMolecule.from_file(str(output_pdbqt), skip_typing=True)
        rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
        if rdkit_mols:
            docked_mol = rdkit_mols[0]

    output = {
        "smiles": smiles,
        "best_affinity_kcal_mol": best_affinity,
        "docked_pdbqt": str(output_pdbqt),
        "from_cache": False,
    }

    if use_cache:
        import json

        cache_file.write_text(json.dumps({k: v for k, v in output.items() if k != "docked_mol"}))

    output["docked_mol"] = docked_mol
    return output


def dock_batch(
    smiles_list: list[str],
    receptor_pdbqt: Path,
    center: tuple[float, float, float],
    box_size: tuple[float, float, float] = (20.0, 20.0, 20.0),
    exhaustiveness: int = 8,
    max_workers: int = 4,
    use_cache: bool = True,
) -> list[dict]:
    """Dock a queue of candidates (e.g. output of the generative model,
    see TODO/ml/TODO_generative_model.md) against one receptor in
    parallel. Threads, not processes: each Vina call is a subprocess, so
    the GIL is released while waiting on it and threads are enough to get
    real parallelism without the pickling overhead of a process pool.
    A failure on one molecule is caught and reported per-molecule rather
    than aborting the whole batch.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[Optional[dict]] = [None] * len(smiles_list)

    def _dock_one(i: int, smiles: str) -> tuple[int, dict]:
        try:
            r = dock(smiles, receptor_pdbqt, center, box_size, exhaustiveness, seed=i, use_cache=use_cache)
            return i, r
        except Exception as exc:
            return i, {"smiles": smiles, "best_affinity_kcal_mol": None, "error": str(exc)}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_dock_one, i, s) for i, s in enumerate(smiles_list)]
        for future in as_completed(futures):
            i, r = future.result()
            results[i] = r

    return results


def _parse_best_affinity(vina_stdout: str) -> Optional[float]:
    """Vina prints a results table like:
       mode |   affinity | dist from best mode
            | (kcal/mol) | rmsd l.b.| rmsd u.b.
       -----+------------+----------+----------
          1       -4.2          0          0
    We want the affinity of mode 1 (the best pose).
    """
    lines = vina_stdout.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("1"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    continue
    return None

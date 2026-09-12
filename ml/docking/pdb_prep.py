"""Split a raw PDB file (as downloaded from RCSB) into a protein-only
receptor file and a single ligand's coordinates, and compute the ligand's
centroid as the docking box center. See
TODO/data/TODO_preprocessing_pipeline.md ("binding pocket з PDBbind-
метаданих") — here applied to a standalone public PDB structure rather
than PDBbind, since PDBbind itself requires manual registration (see
DATA_SOURCES.md) and was not obtained.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem


def split_receptor_ligand(pdb_path: Path, ligand_resname: str, out_dir: Path) -> dict:
    """Write `<out_dir>/receptor.pdb` (protein ATOM records only, waters
    and all HETATM removed) and `<out_dir>/ligand.pdb` (just the named
    ligand's HETATM records), and return the ligand's centroid.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    receptor_lines, ligand_lines, ligand_coords = [], [], []

    for line in pdb_path.read_text().splitlines():
        record = line[:6].strip()
        if record == "ATOM":
            receptor_lines.append(line)
        elif record == "HETATM":
            resname = line[17:20].strip()
            if resname == ligand_resname:
                ligand_lines.append(line)
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                ligand_coords.append((x, y, z))
        elif record in ("TER", "END"):
            receptor_lines.append(line)

    if not ligand_lines:
        raise ValueError(f"No HETATM records found for ligand residue {ligand_resname!r} in {pdb_path}")

    receptor_path = out_dir / "receptor.pdb"
    ligand_path = out_dir / "ligand.pdb"
    receptor_path.write_text("\n".join(receptor_lines) + "\n")
    ligand_path.write_text("\n".join(ligand_lines) + "\nEND\n")

    centroid = np.mean(np.array(ligand_coords), axis=0)
    return {
        "receptor_pdb": receptor_path,
        "ligand_pdb": ligand_path,
        "box_center": tuple(centroid.tolist()),
        "n_ligand_atoms": len(ligand_coords),
    }


def ligand_mol_from_pdb(ligand_pdb_path: Path):
    """Load the crystal ligand as an RDKit Mol (bond perception from
    3D distances) — used to get its canonical SMILES and, separately, as
    the reference structure for redocking RMSD."""
    mol = Chem.MolFromPDBFile(str(ligand_pdb_path), sanitize=True, removeHs=True)
    if mol is None:
        raise ValueError(f"RDKit failed to parse ligand from {ligand_pdb_path}")
    return mol

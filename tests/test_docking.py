"""Docking tests run against a real public structure (3PTB: trypsin +
benzamidine, from RCSB PDB, public domain) rather than mocks, since the
whole point of ml/TODO_docking_scoring.md is validating that the wrapper
produces sane real docking results. Skipped automatically if the Vina
executable or receptor-prep tooling isn't available in this environment
(see scripts/download_vina.py) rather than failing CI hard on a missing
third-party binary.
"""
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from ml.docking.pdb_prep import ligand_mol_from_pdb, split_receptor_ligand
from ml.docking.vina_wrapper import VINA_EXE, dock, dock_batch, prepare_receptor_pdbqt

FIXTURE_PDB = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "docking" / "3ptb.pdb"
SPLIT_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "docking" / "test_split"

requires_vina = pytest.mark.skipif(not VINA_EXE.exists(), reason="Vina executable not installed (see scripts/download_vina.py)")


@pytest.fixture(scope="module")
def receptor_setup():
    split = split_receptor_ligand(FIXTURE_PDB, "BEN", SPLIT_DIR)
    receptor_pdbqt = prepare_receptor_pdbqt(split["receptor_pdb"], SPLIT_DIR / "receptor")
    return {**split, "receptor_pdbqt": receptor_pdbqt}


def test_split_receptor_ligand_finds_benzamidine():
    split = split_receptor_ligand(FIXTURE_PDB, "BEN", SPLIT_DIR)
    assert split["n_ligand_atoms"] == 9  # benzamidine has 9 heavy atoms
    assert split["ligand_pdb"].exists()
    assert split["receptor_pdb"].exists()


def test_split_receptor_ligand_rejects_unknown_resname():
    with pytest.raises(ValueError):
        split_receptor_ligand(FIXTURE_PDB, "NOT_A_REAL_LIGAND", SPLIT_DIR)


def test_ligand_mol_from_pdb_has_correct_connectivity():
    split = split_receptor_ligand(FIXTURE_PDB, "BEN", SPLIT_DIR)
    native_raw = ligand_mol_from_pdb(split["ligand_pdb"])
    reference = Chem.MolFromSmiles("NC(=N)c1ccccc1")
    # Bond orders from raw PDB distance-perception are often wrong (this is
    # documented, not a bug) but connectivity/atom count must match the
    # known reference for AssignBondOrdersFromTemplate to succeed at all.
    fixed = AllChem.AssignBondOrdersFromTemplate(reference, native_raw)
    assert fixed.GetNumAtoms() == reference.GetNumAtoms()


@requires_vina
def test_prepare_receptor_pdbqt(receptor_setup):
    assert receptor_setup["receptor_pdbqt"].exists()
    assert receptor_setup["receptor_pdbqt"].stat().st_size > 0


@requires_vina
def test_dock_benzamidine_against_trypsin_redocking(receptor_setup):
    """The core validation this file exists for: redock the known native
    ligand and check both the affinity estimate and pose RMSD are
    plausible, against the real receptor structure — not a synthetic
    sanity check."""
    result = dock(
        smiles="NC(=N)c1ccccc1",
        receptor_pdbqt=receptor_setup["receptor_pdbqt"],
        center=receptor_setup["box_center"],
        box_size=(20, 20, 20),
        exhaustiveness=8,
        use_cache=False,
    )
    assert result["best_affinity_kcal_mol"] is not None
    assert result["best_affinity_kcal_mol"] < 0  # favorable binding

    native_raw = ligand_mol_from_pdb(receptor_setup["ligand_pdb"])
    reference = Chem.MolFromSmiles("NC(=N)c1ccccc1")
    native_mol = Chem.RemoveHs(AllChem.AssignBondOrdersFromTemplate(reference, native_raw))
    docked_mol = Chem.RemoveHs(result["docked_mol"])

    rmsd = rdMolAlign.GetBestRMS(docked_mol, native_mol)
    assert rmsd < 2.0  # standard "successful redocking" threshold in the field


@requires_vina
def test_dock_batch_orders_results_correctly(receptor_setup):
    molecules = ["NC(=N)c1ccccc1", "CCO"]  # known binder vs. a weak/non-binder sanity check
    results = dock_batch(
        molecules, receptor_setup["receptor_pdbqt"], receptor_setup["box_center"],
        box_size=(20, 20, 20), exhaustiveness=4, max_workers=2, use_cache=False,
    )
    assert [r["smiles"] for r in results] == molecules
    # benzamidine (the actual co-crystallized inhibitor) should bind more
    # favorably than ethanol against this pocket.
    assert results[0]["best_affinity_kcal_mol"] < results[1]["best_affinity_kcal_mol"]


@requires_vina
def test_dock_batch_reports_error_without_aborting_whole_batch(receptor_setup):
    results = dock_batch(
        ["not a valid smiles", "CCO"], receptor_setup["receptor_pdbqt"], receptor_setup["box_center"],
        box_size=(20, 20, 20), exhaustiveness=4, max_workers=2, use_cache=False,
    )
    assert results[0]["best_affinity_kcal_mol"] is None
    assert "error" in results[0]
    assert results[1]["best_affinity_kcal_mol"] is not None

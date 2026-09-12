from ml.data.splits import dedup_smiles, murcko_scaffold, scaffold_split


def test_dedup_smiles_removes_exact_and_equivalent_duplicates():
    smiles = ["CCO", "OCC", "CCN", "invalid(((", "CCO"]
    # "CCO" and "OCC" are the same molecule (ethanol) written differently;
    # the second "CCO" is an exact duplicate; the invalid entry is dropped.
    keep = dedup_smiles(smiles)
    kept_smiles = [smiles[i] for i in keep]
    assert kept_smiles == ["CCO", "CCN"]


def test_murcko_scaffold_shared_by_ring_analogs():
    # Toluene and chlorobenzene share a benzene scaffold.
    s1 = murcko_scaffold("Cc1ccccc1")
    s2 = murcko_scaffold("Clc1ccccc1")
    assert s1 == s2 == "c1ccccc1"


def test_murcko_scaffold_invalid_returns_none():
    assert murcko_scaffold("not a molecule") is None


def test_scaffold_split_no_overlap_and_full_coverage():
    smiles = [
        "Cc1ccccc1",       # benzene scaffold
        "Clc1ccccc1",      # benzene scaffold
        "Fc1ccccc1",       # benzene scaffold
        "CCCCCC",          # hexane, no ring -> empty scaffold
        "CCCCCCC",         # heptane, no ring -> empty scaffold
        "C1CCCCC1",        # cyclohexane scaffold
        "C1CCCCC1C",       # methylcyclohexane -> different scaffold size
    ]
    train, valid, test = scaffold_split(smiles, frac_train=0.6, frac_valid=0.2, frac_test=0.2)

    all_idx = train + valid + test
    assert sorted(all_idx) == list(range(len(smiles)))  # every molecule assigned exactly once
    assert set(train).isdisjoint(valid)
    assert set(train).isdisjoint(test)
    assert set(valid).isdisjoint(test)

    # The three benzene-scaffold molecules must land on the same side of the split.
    benzene_indices = {0, 1, 2}
    for split in (train, valid, test):
        overlap = benzene_indices & set(split)
        assert overlap in (set(), benzene_indices)


def test_scaffold_split_drops_invalid_smiles():
    smiles = ["CCO", "not valid", "CCN"]
    train, valid, test = scaffold_split(smiles, frac_train=1.0, frac_valid=0.0, frac_test=0.0)
    assert 1 not in (train + valid + test)

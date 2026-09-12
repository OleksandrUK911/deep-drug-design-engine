# Data Sources

This project uses only public datasets. Raw and processed data files are
**not** committed to this repository (see `.gitignore`) — run
`scripts/download_datasets.py` to fetch them locally. Small fixture samples
(first 20 rows of each dataset) live in `data/fixtures/` for tests and are
committed.

Checksums below were recorded from a verified download on **2026-09-12**.
If a re-download produces a different checksum, treat the source as
untrusted until manually confirmed — do not silently update the recorded
hash.

## Property prediction (GNN fine-tuning)

| Dataset | Task | License / provenance | File | Size | SHA-256 |
|---|---|---|---|---|---|
| [Tox21](https://tripod.nih.gov/tox21/challenge/) | 12-task toxicity classification | Public domain (NIH/EPA Tox21 Data Challenge); redistributed via [MoleculeNet](https://moleculenet.org/) | `tox21.csv.gz` | 122,925 B | `45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360` |
| [BBBP](https://moleculenet.org/datasets-1) | Blood-brain barrier penetration (binary) | Academic, redistributed via MoleculeNet for research use | `BBBP.csv` | 148,743 B | `d07a38487aeac5cee5508413e468043ef3097451d2a112701c2d60be9ec6b662` |
| [ESOL (Delaney)](https://moleculenet.org/datasets-1) | Aqueous solubility (regression) | Academic (Delaney, 2004), redistributed via MoleculeNet | `delaney-processed.csv` | 96,699 B | `8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189` |

Source host for all three: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/` (DeepChem's public MoleculeNet mirror). Used for research/educational fine-tuning only, with attribution to the original dataset authors.

## SSL pretraining + generative model training

| Dataset | Purpose | License / provenance | File | Size | SHA-256 |
|---|---|---|---|---|---|
| [ZINC250k](https://zinc15.docking.org/) | Self-supervised pretraining corpus + generative model training | ZINC is free to use for research; this specific 250k curated subset (with precomputed logP/QED/SAS) is the version popularized by [Gómez-Bombarelli et al., 2018](https://github.com/aspuru-guzik-group/chemical_vae) and widely reused in generative-chemistry papers | `zinc250k.csv` | 22,606,589 B | `35e3f1a52b1badc0697e373d73a18ad773f415936ff992f4c6baa2e067b3e6ae` |

Source: `https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv`

**Commercial use note:** ZINC's bulk/commercial terms differ from browsing terms — this project only ever uses ZINC250k for non-commercial research/educational model training, never for a commercial product.

## Docking (classical + generative)

| Dataset | Purpose | License / provenance | Access |
|---|---|---|---|
| [PDBbind](http://www.pdbbind.org.cn/) (refined set) | Ligand-receptor complexes with known binding pose/affinity — training and evaluation for `ml/TODO_docking_scoring.md` and `ml/TODO_generative_docking.md` | Free academic license, but requires creating an account and accepting PDBbind's license agreement — **not scriptable** | Manual: register at pdbbind.org.cn → request the "refined set" → download manually into `data/raw/pdbbind/` (gitignored) |

### Demo targets (2-3 receptor/pocket pairs for public demo)
Not yet selected — tracked as a P1 item in `TODO/data/TODO_sources_licensing.md`. Candidates will come from the RCSB PDB (public domain / CC0 structures) with explicit attribution to the depositing structure's authors.

### Protein sequence/embedding data (for `ml/TODO_target_conditioning.md`)
Sequences for the demo targets above will be pulled from [UniProt](https://www.uniprot.org/) (public, CC BY 4.0) — tracked in `TODO/data/TODO_protein_target_data.md`.

## How to (re)download

```bash
python scripts/download_datasets.py           # download + verify all scriptable sources
python scripts/download_datasets.py --only zinc250k
python scripts/download_datasets.py --verify-only   # re-check existing files without re-downloading
```

PDBbind must be obtained manually per the steps above; the script does not attempt it.

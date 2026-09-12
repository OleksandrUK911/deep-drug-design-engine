#!/usr/bin/env python3
"""Download and checksum-verify the public datasets used by this project.

Datasets and licensing are documented in DATA_SOURCES.md. This script only
handles the sources that can be fetched anonymously over HTTP (MoleculeNet
property datasets, ZINC250k). PDBbind requires a free account and a manual
license click-through, so it is not scriptable — see DATA_SOURCES.md for the
manual steps.

Usage:
    python scripts/download_datasets.py [--only NAME] [--verify-only]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"


@dataclass(frozen=True)
class DatasetSource:
    name: str
    url: str
    filename: str
    sha256: str
    note: str


# Checksums recorded from a verified download on 2026-09-12 — see DATA_SOURCES.md.
DATASETS: list[DatasetSource] = [
    DatasetSource(
        name="tox21",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        filename="tox21.csv.gz",
        sha256="45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360",
        note="MoleculeNet Tox21 — 12-task toxicity classification (fine-tuning target).",
    ),
    DatasetSource(
        name="bbbp",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        filename="BBBP.csv",
        sha256="d07a38487aeac5cee5508413e468043ef3097451d2a112701c2d60be9ec6b662",
        note="MoleculeNet BBBP — blood-brain barrier penetration (binary classification).",
    ),
    DatasetSource(
        name="esol",
        url="https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        filename="delaney-processed.csv",
        sha256="8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189",
        note="MoleculeNet ESOL (Delaney) — aqueous solubility regression.",
    ),
    DatasetSource(
        name="zinc250k",
        url=(
            "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/"
            "master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
        ),
        filename="zinc250k.csv",
        sha256="35e3f1a52b1badc0697e373d73a18ad773f415936ff992f4c6baa2e067b3e6ae",
        note="ZINC250k subset — SSL pretraining + generative model training corpus.",
    ),
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(source: DatasetSource, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source.filename
    print(f"[{source.name}] downloading -> {dest}")
    urllib.request.urlretrieve(source.url, dest)
    return dest


def verify(source: DatasetSource, dest: Path) -> bool:
    if not dest.exists():
        print(f"[{source.name}] MISSING: {dest}")
        return False
    digest = sha256_of(dest)
    ok = digest == source.sha256
    status = "OK" if ok else "CHECKSUM MISMATCH"
    print(f"[{source.name}] {status} sha256={digest}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="download/verify only this dataset name")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="skip downloading, only verify checksums of files already present",
    )
    args = parser.parse_args()

    datasets = [d for d in DATASETS if args.only in (None, d.name)]
    if not datasets:
        print(f"Unknown dataset name: {args.only}", file=sys.stderr)
        return 1

    all_ok = True
    for source in datasets:
        dest = RAW_DIR / source.filename
        if not args.verify_only:
            download(source, RAW_DIR)
        all_ok = verify(source, dest) and all_ok

    if not all_ok:
        print(
            "\nOne or more checksums did not match the recorded value in "
            "DATA_SOURCES.md. Do not train on unverified data — re-download "
            "or update the recorded checksum only after manually confirming "
            "the new source is legitimate.",
            file=sys.stderr,
        )
        return 1

    print("\nAll requested datasets downloaded and verified.")
    print(
        "PDBbind is not included here — it requires a free account and "
        "manual license acceptance. See DATA_SOURCES.md for the steps."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

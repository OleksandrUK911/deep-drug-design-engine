#!/usr/bin/env python3
"""Download the standalone AutoDock Vina executable for the current
platform into bin/ (gitignored — it's a third-party binary, not project
code). See TODO/ml/TODO_docking_scoring.md.

The `vina` PyPI package requires compiling against Boost and has no
prebuilt wheel for many platform/Python combinations (confirmed failing
here on Windows/Python 3.13) — the standalone executable, invoked via
subprocess in ml/docking/vina_wrapper.py, is the officially distributed
alternative and produces identical scores.
"""
from __future__ import annotations

import platform
import stat
import sys
import urllib.request
from pathlib import Path

VERSION = "1.2.7"
BASE_URL = f"https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v{VERSION}/"

ASSET_BY_PLATFORM = {
    ("Windows", "AMD64"): "vina_1.2.7_win.exe",
    ("Linux", "x86_64"): "vina_1.2.7_linux_x86_64",
    ("Linux", "aarch64"): "vina_1.2.7_linux_aarch64",
    ("Darwin", "x86_64"): "vina_1.2.7_mac_x86_64",
    ("Darwin", "arm64"): "vina_1.2.7_mac_aarch64",
}


def main() -> int:
    key = (platform.system(), platform.machine())
    asset = ASSET_BY_PLATFORM.get(key)
    if asset is None:
        print(f"No known Vina {VERSION} release asset for platform {key}.", file=sys.stderr)
        print(f"Check https://github.com/ccsb-scripps/AutoDock-Vina/releases/tag/v{VERSION} manually.", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parent.parent
    bin_dir = repo_root / "bin"
    bin_dir.mkdir(exist_ok=True)
    dest = bin_dir / ("vina.exe" if key[0] == "Windows" else "vina")

    url = BASE_URL + asset
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)

    if key[0] != "Windows":
        dest.chmod(dest.stat().st_mode | stat.S_IEXEC)

    print(f"Done. Vina {VERSION} installed at {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

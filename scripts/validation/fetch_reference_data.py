#!/usr/bin/env python3
"""
Script: scripts/validation/fetch_reference_data.py
Purpose: Re-download every NASA TMR reference file referenced by the regime
         metadata into validation_data/. Computes a SHA-256 for each file and
         writes validation_data/manifest.json with download timestamps and
         hashes for provenance.

Usage:
    micromamba run -n openfoam python scripts/validation/fetch_reference_data.py
    micromamba run -n openfoam python scripts/validation/fetch_reference_data.py --check  # verify only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VALIDATION_DATA_DIR = PROJECT_ROOT / "validation_data"
MANIFEST_PATH = VALIDATION_DATA_DIR / "manifest.json"

TMR_NACA0012_BASE = "https://tmbwg.github.io/turbmodels/NACA0012_validation"
TMR_NACA4412_BASE = "https://tmbwg.github.io/turbmodels/NACA4412sep_validation"

# Mapping of local relative path → upstream URL
DOWNLOADS: dict[str, str] = {
    # Experimental — shared by Regimes A, B, D
    "common/CLCD_Ladson_expdata.dat":      f"{TMR_NACA0012_BASE}/CLCD_Ladson_expdata.dat",
    "common/CP_Ladson.dat":                f"{TMR_NACA0012_BASE}/CP_Ladson.dat",
    "common/CL_Gregory_expdata.dat":       f"{TMR_NACA0012_BASE}/CL_Gregory_expdata.dat",
    "common/CP_Gregory_expdata.dat":       f"{TMR_NACA0012_BASE}/CP_Gregory_expdata.dat",
    "common/0012.abbottdata.cl.dat":       f"{TMR_NACA0012_BASE}/0012.abbottdata.cl.dat",
    "common/0012.abbottdata.cd.dat":       f"{TMR_NACA0012_BASE}/0012.abbottdata.cd.dat",
    "common/0012.mccroskeydata.cl.dat":    f"{TMR_NACA0012_BASE}/0012.mccroskeydata.cl.dat",
    # CFD reference — Spalart-Allmaras
    "common/n0012clcd_cfl3d_sa.dat":       f"{TMR_NACA0012_BASE}/n0012clcd_cfl3d_sa.dat",
    "common/n0012cp_cfl3d_sa.dat":         f"{TMR_NACA0012_BASE}/n0012cp_cfl3d_sa.dat",
    "common/n0012cf_cfl3d_sa.dat":         f"{TMR_NACA0012_BASE}/n0012cf_cfl3d_sa.dat",
    # CFD reference — k-ω SST
    "common/n0012clcd_cfl3d_sst.dat":      f"{TMR_NACA0012_BASE}/n0012clcd_cfl3d_sst.dat",
    "common/n0012cp_cfl3d_sst.dat":        f"{TMR_NACA0012_BASE}/n0012cp_cfl3d_sst.dat",
    "common/n0012cf_cfl3d_sst.dat":        f"{TMR_NACA0012_BASE}/n0012cf_cfl3d_sst.dat",
    # NACA 4412 separation — Regime B
    "regime_B/raw/naca4412.cp.expt.dat":   f"{TMR_NACA4412_BASE}/naca4412.cp.expt.dat",
    "regime_B/raw/exp.profiles.new.dat":   f"{TMR_NACA4412_BASE}/exp.profiles.new.dat",
}

USER_AGENT = "NACASurrogate-validation-fetcher/1.0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download NASA TMR reference data for NACASurrogate validation.")
    p.add_argument("--check", action="store_true",
                   help="Do not download; only verify presence and recompute hashes.")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if the local file already exists.")
    return p.parse_args()


def download(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and not force:
        log.info("present  %s", dest.relative_to(PROJECT_ROOT))
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as fh:
            fh.write(resp.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} fetching {url}") from exc
    log.info("fetched  %s  (%d bytes)", dest.relative_to(PROJECT_ROOT), dest.stat().st_size)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest() -> dict:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": {},
    }
    for relpath, url in sorted(DOWNLOADS.items()):
        local = VALIDATION_DATA_DIR / relpath
        if not local.exists():
            manifest["files"][relpath] = {
                "url": url, "status": "missing", "sha256": None, "size": None,
            }
            continue
        manifest["files"][relpath] = {
            "url": url,
            "status": "present",
            "sha256": sha256(local),
            "size": local.stat().st_size,
        }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log.info("wrote manifest → %s", MANIFEST_PATH.relative_to(PROJECT_ROOT))
    return manifest


def main() -> int:
    args = parse_args()
    if not args.check:
        for relpath, url in DOWNLOADS.items():
            download(url, VALIDATION_DATA_DIR / relpath, force=args.force)
    manifest = write_manifest()
    missing = [f for f, m in manifest["files"].items() if m["status"] == "missing"]
    if missing:
        log.error("missing %d files: %s", len(missing), ", ".join(missing))
        return 1
    log.info("all %d reference files present and hashed", len(manifest["files"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Script: scripts/airfrans/inspect_dataset.py
Stage:  AirfRANS Phase 2 — dataset inspection (Gate 2)
Purpose: Load the local AirfRANS dataset (default: the clipped variant), log
         what each sample holds, and check that everything the coefficient and
         surface-curve surrogates need is present.

Writes results/airfrans_inspection.md. Exits non-zero if any Gate 2 check fails.

Usage:
    uv run python scripts/airfrans/inspect_dataset.py [--data-dir DIR] [--reference-dir DIR]
"""

from __future__ import annotations

import argparse
import logging
import resource
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.airfrans import io  # noqa: E402
from scripts.airfrans.geometry import WALL_TOL, extract_surface, wall_node_mask  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_PATH = RESULTS_DIR / "airfrans_inspection.md"
N_DETAILED = 3
REFERENCE_DIR = PROJECT_ROOT / "data" / "airfrans_remeshed"   # the variant the splits were first frozen from


def describe_sample(i: int, sample) -> list[str]:
    lines = [f"### Sample {i}", ""]
    lines.append("| scalar | value |")
    lines.append("|---|---|")
    for name in sample.get_scalar_names():
        lines.append(f"| `{name}` | {float(sample.get_scalar(name)):.6g} |")
    nodes = sample.get_nodes()
    elements = sample.get_elements()
    lines += [
        "",
        f"- fields: {', '.join(f'`{f}`' for f in sample.get_field_names())}",
        f"- nodes: {nodes.shape[0]}; elements: "
        + ", ".join(f"{k} × {len(v)}" for k, v in elements.items()),
        f"- bases: {sample.get_base_names()}; zones: {sample.get_zone_names()}",
        f"- nodal tags: {list(sample.get_nodal_tags().keys()) or 'none'}",
        f"- node bounding box: x ∈ [{nodes[:, 0].min():.3f}, {nodes[:, 0].max():.3f}], "
        f"y ∈ [{nodes[:, 1].min():.3f}, {nodes[:, 1].max():.3f}]",
        "",
    ]
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--reference-dir", default=str(REFERENCE_DIR),
                    help="another AirfRANS variant whose card splits must match (skipped if absent)")
    args = ap.parse_args()

    data_dir = io.resolve_data_dir(args.data_dir)
    log.info("Reading AirfRANS from %s", data_dir)
    card = io.load_card(data_dir)
    splits = io.load_splits(data_dir)
    reader = io.RawSampleReader(data_dir)
    n = len(reader)

    md = [f"# {card.get('pretty_name', 'AirfRANS')} — dataset inspection (Gate 2)", "",
          f"Data directory: `{data_dir.relative_to(PROJECT_ROOT) if data_dir.is_relative_to(PROJECT_ROOT) else data_dir}`",
          f"Parquet shards: {len(io.shard_paths(data_dir))}; samples: {n}", "",
          "## Card metadata", ""]
    desc = card["dataset_info"]["description"]
    for key in ("in_scalars_names", "out_scalars_names", "in_fields_names",
                "out_fields_names", "in_meshes_names"):
        md.append(f"- `{key}`: {desc.get(key)}")
    md += ["", "## Splits (from the card; row indices into `all_samples`)", "",
           "| split | n | min idx | max idx |", "|---|---|---|---|"]
    for name, ids in splits.items():
        md.append(f"| `{name}` | {len(ids)} | {ids.min()} | {ids.max()} |")
    md.append("")

    # --- detailed view of the first few samples ---------------------------
    md += ["## First samples", ""]
    scalar_names: set[str] = set()
    for i in range(N_DETAILED):
        sample = io.deserialise(reader.raw_bytes(i))
        md += describe_sample(i, sample)
        scalar_names |= {str(s) for s in sample.get_scalar_names()}

    # --- aggregate pass over every sample ---------------------------------
    rows = []
    field_names: set[str] = set()
    all_triangles = True
    t0 = time.perf_counter()
    for i in tqdm(range(n), desc="scanning samples"):
        sample = io.deserialise(reader.raw_bytes(i))
        field_names |= {str(f) for f in sample.get_field_names()} if i == 0 else set()
        all_triangles &= all("TRI" in str(k).upper() for k in sample.get_elements())
        d = io.extract(sample, i)
        wall = wall_node_mask(d)
        extract_surface(d)  # raises unless the wall is one closed loop
        rows.append((d.scalars.get(io.SCALAR_AOA, np.nan), d.scalars.get(io.SCALAR_UINF, np.nan),
                     d.scalars.get(io.SCALAR_CL, np.nan), d.scalars.get(io.SCALAR_CD, np.nan),
                     len(d.x), int(wall.sum()), float(np.abs(d.implicit_distance[wall]).max())))
    seconds_per_sample = (time.perf_counter() - t0) / n
    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # Linux reports KiB
    arr = np.array(rows, dtype=np.float64)
    aoa, uinf, cl, cd, nnodes, nwall, wall_dmax = arr.T
    re = io.reynolds(uinf)
    aoa_is_rad = np.nanmax(np.abs(aoa)) < 1.0

    md += ["## Aggregate over all samples", "",
           "| quantity | min | median | max |", "|---|---|---|---|"]
    for label, v in [("angle_of_attack (raw)", aoa),
                     ("α (deg)", np.degrees(aoa) if aoa_is_rad else aoa),
                     ("inlet_velocity (m/s)", uinf), ("Re = U∞·c/ν", re),
                     ("C_L", cl), ("C_D", cd), ("mesh nodes", nnodes),
                     ("wall nodes (boundary loop off the clip box)", nwall),
                     ("max |implicit_distance| on wall nodes", wall_dmax)]:
        md.append(f"| {label} | {np.nanmin(v):.6g} | {np.nanmedian(v):.6g} | {np.nanmax(v):.6g} |")
    md += ["", f"ν(T = {io.TEMPERATURE} K) = {io.NU:.6e} m²/s (AirfRANS polynomial); c = {io.CHORD} m.",
           "", "Per-sample identifier: the samples carry no original AirfRANS simulation name "
           "(no name scalar, no tag, no metadata field); `sample_id` is the row index into "
           "`all_samples`, which is also what the card's split lists index.", "",
           f"Streaming cost: {seconds_per_sample:.2f} s per sample (read + deserialise + wall "
           f"extraction), peak RSS {peak_rss_gb:.2f} GB.", ""]

    reference_dir = Path(args.reference_dir).resolve()
    if reference_dir != data_dir and (reference_dir / "README.md").exists():
        ref = io.load_splits(reference_dir)
        splits_match = (sorted(ref) == sorted(splits)
                        and all(np.array_equal(ref[k], splits[k]) for k in splits))
        md += [f"Split lists compared with `{reference_dir.name}`: "
               f"{'identical' if splits_match else 'DIFFERENT'}.", ""]
    else:
        splits_match = None

    # --- Gate 2 -------------------------------------------------------------
    checks = [
        ("C_L and C_D are per-sample scalars",
         {io.SCALAR_CL, io.SCALAR_CD} <= scalar_names and np.isfinite(cl).all() and np.isfinite(cd).all()),
        ("angle of attack and inlet velocity present",
         {io.SCALAR_AOA, io.SCALAR_UINF} <= scalar_names and np.isfinite(aoa).all() and np.isfinite(uinf).all()),
        ("AoA units are radians, range ≈ [−0.0873, 0.2618]",
         aoa_is_rad and aoa.min() > -0.0873 - 1e-3 and aoa.max() < 0.2618 + 1e-3),
        ("derived Re within ≈ [2e6, 6e6]", re.min() > 1.9e6 and re.max() < 6.1e6),
        ("aerofoil wall identifiable (single closed boundary loop off the clip box, ≥ 50 nodes, "
         f"all with |implicit_distance| < {WALL_TOL:g})",
         bool((nwall >= 50).all()) and bool((wall_dmax < WALL_TOL).all())),
        (f"{io.EXPECTED_SAMPLES} samples", n == io.EXPECTED_SAMPLES),
        ("flow fields present (" + ", ".join(io.FLOW_FIELDS) + ")", set(io.FLOW_FIELDS) <= field_names),
        ("every mesh is triangles only", all_triangles),
    ]
    if splits_match is not None:
        checks.append((f"card splits identical to {reference_dir.name}", splits_match))
    md += ["## Gate 2", ""]
    for label, ok in checks:
        md.append(f"- [{'x' if ok else ' '}] {label}")
        (log.info if ok else log.error)("Gate 2 — %s: %s", label, "PASS" if ok else "FAIL")
    passed = all(ok for _, ok in checks)
    md += ["", f"**Gate 2: {'PASS' if passed else 'FAIL'}**", ""]

    RESULTS_DIR.mkdir(exist_ok=True)
    REPORT_PATH.write_text("\n".join(md))
    log.info("Wrote %s", REPORT_PATH.relative_to(PROJECT_ROOT))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

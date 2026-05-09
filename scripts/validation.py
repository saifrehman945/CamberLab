#!/usr/bin/env python3
"""
Script: validation.py
Stage:  Validation — single-point CFD vs published reference data
Purpose: Build one OpenFOAM case at a known (alpha, Re, thickness) reference
         point, run it through the existing meshing and solver pipeline, then
         compare the resulting Cl / Cd against published experimental values
         (Ladson, NASA TM-4074, NACA 0012, Re = 6e6, trip-free model) and
         thin-airfoil theory. Writes a comparison plot and a summary CSV.

Reference:
    Ladson, C. L., "Effects of Independent Variation of Mach and Reynolds
    Numbers on the Low-Speed Aerodynamic Characteristics of the NACA 0012
    Airfoil Section", NASA TM-4074, 1988. Tabulated values reproduced from
    NASA Langley Turbulence Modeling Resource:
    https://turbmodels.larc.nasa.gov/naca0012_val.html

Usage:
    micromamba run -n openfoam python scripts/validation.py
    micromamba run -n openfoam python scripts/validation.py --alpha 4.04
    micromamba run -n openfoam python scripts/validation.py --alpha 0.04 4.04 10.12
    micromamba run -n openfoam python scripts/validation.py --skip-run    # plot only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CASES_DIR = PROJECT_ROOT / "cases"
VALIDATION_DIR = PROJECT_ROOT / "validation"

# Ladson NASA TM-4074, NACA 0012, Re = 6e6, transition fixed at x/c = 0.05.
# This is the canonical reference for fully-turbulent RANS validation
# (kOmegaSST has no transition model, so the trip-free Ladson set would
# bias Cd low — use the tripped set to match the model's assumption).
# Columns: alpha_deg, Cl, Cd
LADSON_NACA0012_RE6E6 = np.array(
    [
        [-3.99, -0.4363, 0.00871],
        [-1.98, -0.2213, 0.00792],
        [-0.03, -0.0115, 0.00803],
        [0.04, -0.0013, 0.00811],
        [2.00, 0.2213, 0.00814],
        [4.06, 0.4365, 0.00814],
        [6.09, 0.6558, 0.00851],
        [8.09, 0.8689, 0.00985],
        [10.18, 1.0809, 0.01165],
        [11.13, 1.1731, 0.01247],
        [12.10, 1.2644, 0.01299],
        [13.31, 1.3676, 0.01408],
        [14.08, 1.4316, 0.01533],
        [15.24, 1.5169, 0.01870],
        [16.33, 1.5855, 0.02186],
        [17.13, 1.6219, 0.02513],
        [18.21, 1.0104, 0.25899],
        [19.27, 1.0664, 0.43446],
    ],
    dtype=np.float64,
)
LADSON_NACA0012_RE6E6_grit_80 = np.array(
    [
        [0.00, 0.0000, 0.00819],
        [4.04, 0.4347, 0.00892],
        [6.09, 0.6438, 0.01016],
        [8.30, 0.8542, 0.01193],
        [10.12, 1.0418, 0.01437],
        [12.13, 1.2356, 0.01776],
        [14.22, 1.4115, 0.02218],
        [15.26, 1.4946, 0.02476],
    ],
    dtype=np.float64,
)
REFERENCE_RE = 6.0e6
REFERENCE_THICKNESS = 0.12
REFERENCE_LABEL = "Ladson NASA TM-4074 (NACA 0012, Re=6e6, transition fixed at x/c=0.05)"
DEFAULT_ALPHAS = (10.12,)

CONVERGENCE_WINDOW = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one or more validation CFD cases against published NACA 0012 "
            "reference data and produce a comparison plot."
        )
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHAS),
        help=(
            "Angle(s) of attack in degrees. Each value should match a row in "
            "the embedded Ladson table (within --alpha-tol). "
            f"Default: {DEFAULT_ALPHAS[0]} deg."
        ),
    )
    parser.add_argument(
        "--alpha-tol",
        type=float,
        default=0.5,
        help="Tolerance (deg) for matching --alpha to a Ladson row. Default 0.5.",
    )
    parser.add_argument(
        "--re",
        type=float,
        default=REFERENCE_RE,
        help=f"Reynolds number for the validation run. Default {REFERENCE_RE:.0e}.",
    )
    parser.add_argument(
        "--thickness",
        type=float,
        default=REFERENCE_THICKNESS,
        help=f"NACA 4-digit thickness ratio. Default {REFERENCE_THICKNESS} (NACA 0012).",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip mesh/solver execution; only post-process and plot existing cases.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild mesh and re-run the solver even if outputs already exist.",
    )
    return parser.parse_args()


def load_module(name: str, source: Path) -> ModuleType:
    """Import a sibling pipeline script whose filename starts with a digit."""
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def lookup_reference(alpha_deg: float, tolerance: float) -> tuple[float, float, float]:
    """Return (alpha_ref, Cl_ref, Cd_ref) from the Ladson table closest to alpha_deg."""
    deltas = np.abs(LADSON_NACA0012_RE6E6[:, 0] - alpha_deg)
    idx = int(np.argmin(deltas))
    if deltas[idx] > tolerance:
        available = ", ".join(f"{a:.2f}" for a in LADSON_NACA0012_RE6E6[:, 0])
        raise ValueError(
            f"alpha={alpha_deg:.3f} deg has no Ladson row within {tolerance:.2f} deg; "
            f"available alphas: [{available}]. Use --alpha-tol to relax."
        )
    return tuple(LADSON_NACA0012_RE6E6[idx].tolist())  # type: ignore[return-value]


def thin_airfoil_cl(alpha_deg: float) -> float:
    """Thin airfoil theory: Cl = 2*pi*alpha (radians) for a symmetric section."""
    return 2.0 * math.pi * math.radians(alpha_deg)


def case_dir_for(alpha_deg: float, reynolds: float, thickness: float) -> Path:
    naca_code = int(round(thickness * 100))
    tag = (
        f"validation_naca{naca_code:04d}"
        f"_re{reynolds:.0e}".replace("+0", "").replace("e0", "e")
        + f"_a{alpha_deg:+06.2f}".replace(".", "p").replace("+", "p").replace("-", "m")
    )
    return CASES_DIR / tag


def write_case_inputs(
    case_dir: Path,
    alpha_deg: float,
    reynolds: float,
    thickness: float,
    geometry: ModuleType,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    coords = geometry.aerofoil_polygon(thickness, geometry.N_POINTS)
    geometry.write_aerofoil_dat(case_dir / "aerofoil.dat", coords)
    geometry.write_params_json(
        case_dir / "params.json",
        alpha_deg=alpha_deg,
        Re=reynolds,
        thickness=thickness,
    )


def build_mesh(case_dir: Path, mesh: ModuleType, force: bool) -> None:
    boundary_path = case_dir / "constant" / "polyMesh" / "boundary"
    if boundary_path.exists() and not force:
        log.info("Mesh already present for %s — skipping (use --force to rebuild)", case_dir.name)
        return

    if force:
        mesh.reset_case_mesh(case_dir)

    gmsh_module = mesh.require_gmsh()
    gmsh_module.initialize()
    try:
        params = mesh.load_params(case_dir)
        metrics = mesh.build_mesh(
            case_dir / "aerofoil.dat",
            params["Re"],
            case_dir,
            chord=mesh.CHORD,
            target_y_plus=mesh.TARGET_Y_PLUS,
        )
    finally:
        gmsh_module.finalize()

    log.info(
        "Meshed %s  cells=%d  nonOrtho=%.2f  skew=%.3f",
        case_dir.name,
        int(metrics["cell_count"]),
        metrics["max_non_orthogonality"],
        metrics["max_skewness"],
    )


def run_solver(case_dir: Path, run_cfd: ModuleType, force: bool) -> None:
    coeff_dat = case_dir / "postProcessing" / "forceCoeffs" / "0" / "forceCoeffs.dat"
    if coeff_dat.exists() and not force:
        log.info("forceCoeffs already present for %s — skipping solver", case_dir.name)
        return

    run_cfd.render_case(case_dir)
    run_cfd.run_case(case_dir)


def parse_force_coeffs(case_dir: Path, window: int = CONVERGENCE_WINDOW) -> dict[str, float]:
    """Average Cl, Cd, Cm over the last `window` rows of forceCoeffs.dat."""
    coeff_path = case_dir / "postProcessing" / "forceCoeffs" / "0" / "forceCoeffs.dat"
    if not coeff_path.exists():
        raise FileNotFoundError(f"{coeff_path} not found — solver did not produce force output")

    df = pd.read_csv(
        coeff_path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["t", "Cm", "Cd", "Cl", "Cl_f", "Cl_r"],
        dtype=np.float64,
        engine="python",
    )

    if len(df) < 50:
        raise RuntimeError(
            f"{coeff_path} has only {len(df)} rows; the solver likely diverged"
        )

    tail = df.tail(min(window, len(df)))
    return {
        "Cl_mean": float(tail["Cl"].mean()),
        "Cl_std": float(tail["Cl"].std()),
        "Cd_mean": float(tail["Cd"].mean()),
        "Cd_std": float(tail["Cd"].std()),
        "Cm_mean": float(tail["Cm"].mean()),
        "n_iters": int(len(df)),
        "n_window": int(len(tail)),
    }


def percent_error(value: float, reference: float) -> float:
    if reference == 0.0:
        return float("nan")
    return 100.0 * (value - reference) / reference


def make_summary(rows: list[dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    column_order = [
        "alpha_deg",
        "Re",
        "thickness",
        "Cl_cfd",
        "Cl_ref",
        "Cl_thin",
        "Cl_err_pct",
        "Cd_cfd",
        "Cd_ref",
        "Cd_err_pct",
        "n_iters",
    ]
    return df[[c for c in column_order if c in df.columns]]


def plot_results(summary: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = summary.sort_values("alpha_deg").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(df) == 1:
        _plot_single_point(df.iloc[0], output_path, plt)
    else:
        _plot_polar(df, output_path, plt)

    log.info("Wrote validation plot → %s", output_path.relative_to(PROJECT_ROOT))


def _plot_single_point(row: pd.Series, output_path: Path, plt) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    cl_labels = ["CFD (foamRun)", "Ladson exp.", "Thin airfoil"]
    cl_values = [row["Cl_cfd"], row["Cl_ref"], row["Cl_thin"]]
    cl_colors = ["#1f77b4", "#2ca02c", "#888888"]
    axes[0].bar(cl_labels, cl_values, color=cl_colors)
    axes[0].set_ylabel("Cl")
    axes[0].set_title(f"Cl  (alpha={row['alpha_deg']:.2f}°,  Re={row['Re']:.1e})")
    for x, v in enumerate(cl_values):
        axes[0].text(x, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    cd_labels = ["CFD (foamRun)", "Ladson exp."]
    cd_values = [row["Cd_cfd"], row["Cd_ref"]]
    cd_colors = ["#1f77b4", "#2ca02c"]
    axes[1].bar(cd_labels, cd_values, color=cd_colors)
    axes[1].set_ylabel("Cd")
    axes[1].set_title(f"Cd  (alpha={row['alpha_deg']:.2f}°,  Re={row['Re']:.1e})")
    for x, v in enumerate(cd_values):
        axes[1].text(x, v, f"{v:.5f}", ha="center", va="bottom", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle(REFERENCE_LABEL, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_polar(df: pd.DataFrame, output_path: Path, plt) -> None:
    alpha_dense = np.linspace(df["alpha_deg"].min() - 1, df["alpha_deg"].max() + 1, 100)
    cl_thin_curve = 2.0 * np.pi * np.radians(alpha_dense)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(alpha_dense, cl_thin_curve, "--", color="#888888", label="Thin airfoil  Cl=2πα")
    axes[0].plot(df["alpha_deg"], df["Cl_ref"], "s-", color="#2ca02c", label="Ladson exp.")
    axes[0].plot(df["alpha_deg"], df["Cl_cfd"], "o-", color="#1f77b4", label="CFD (foamRun)")
    axes[0].set_xlabel("Angle of attack α [deg]")
    axes[0].set_ylabel("Cl")
    axes[0].set_title("Lift coefficient")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)

    axes[1].plot(df["alpha_deg"], df["Cd_ref"], "s-", color="#2ca02c", label="Ladson exp.")
    axes[1].plot(df["alpha_deg"], df["Cd_cfd"], "o-", color="#1f77b4", label="CFD (foamRun)")
    axes[1].set_xlabel("Angle of attack α [deg]")
    axes[1].set_ylabel("Cd")
    axes[1].set_title("Drag coefficient")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best", fontsize=9)

    fig.suptitle(REFERENCE_LABEL, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def report_row(row: dict[str, float]) -> None:
    log.info(
        "α=%5.2f°  Cl: CFD=%.4f  ref=%.4f  thin=%.4f  err=%+6.2f%%  | "
        "Cd: CFD=%.5f  ref=%.5f  err=%+6.2f%%",
        row["alpha_deg"],
        row["Cl_cfd"],
        row["Cl_ref"],
        row["Cl_thin"],
        row["Cl_err_pct"],
        row["Cd_cfd"],
        row["Cd_ref"],
        row["Cd_err_pct"],
    )


def main() -> None:
    args = parse_args()

    geometry = load_module("naca_geometry", SCRIPTS_DIR / "02_geometry.py")
    mesh = load_module("naca_mesh", SCRIPTS_DIR / "03_mesh.py")
    run_cfd = load_module("naca_run_cfd", SCRIPTS_DIR / "04_run_cfd.py")

    rows: list[dict[str, float]] = []
    for alpha_request in args.alpha:
        alpha_ref, cl_ref, cd_ref = lookup_reference(alpha_request, args.alpha_tol)

        case_dir = case_dir_for(alpha_ref, args.re, args.thickness)
        log.info(
            "Validating α=%.2f° (matched ref %.2f°)  Re=%.2e  t=%.4f  → %s",
            alpha_request,
            alpha_ref,
            args.re,
            args.thickness,
            case_dir.relative_to(PROJECT_ROOT),
        )

        if not args.skip_run:
            write_case_inputs(case_dir, alpha_ref, args.re, args.thickness, geometry)
            build_mesh(case_dir, mesh, args.force)
            run_solver(case_dir, run_cfd, args.force)

        params = json.loads((case_dir / "params.json").read_text())
        coeffs = parse_force_coeffs(case_dir)

        row = {
            "alpha_deg": float(params["alpha_deg"]),
            "Re": float(params["Re"]),
            "thickness": float(params["thickness"]),
            "Cl_cfd": coeffs["Cl_mean"],
            "Cl_std": coeffs["Cl_std"],
            "Cd_cfd": coeffs["Cd_mean"],
            "Cd_std": coeffs["Cd_std"],
            "Cl_ref": cl_ref,
            "Cd_ref": cd_ref,
            "Cl_thin": thin_airfoil_cl(alpha_ref),
            "Cl_err_pct": percent_error(coeffs["Cl_mean"], cl_ref),
            "Cd_err_pct": percent_error(coeffs["Cd_mean"], cd_ref),
            "n_iters": coeffs["n_iters"],
            "case_dir": str(case_dir.relative_to(PROJECT_ROOT)),
        }
        rows.append(row)
        report_row(row)

    summary = make_summary(rows)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(VALIDATION_DIR / "validation_summary.csv", index=False)
    log.info("Wrote summary → %s", (VALIDATION_DIR / "validation_summary.csv").relative_to(PROJECT_ROOT))

    plot_path = VALIDATION_DIR / ("validation_polar.png" if len(rows) > 1 else "validation_point.png")
    plot_results(summary, plot_path)


if __name__ == "__main__":
    main()

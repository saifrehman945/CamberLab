#!/usr/bin/env python3
"""
Script: 04_run_cfd.py
Stage:  4 — OpenFOAM case rendering and execution
Purpose: Render the reusable OpenFOAM base case into each design directory and
         optionally run the solver for cases that already contain a mesh.

Usage:
    micromamba run -n openfoam python scripts/04_run_cfd.py
    micromamba run -n openfoam python scripts/04_run_cfd.py --case-id 0 1 2
    micromamba run -n openfoam python scripts/04_run_cfd.py --run --jobs 4

When --run is used with --jobs N > 1, each case is solved sequentially using
N MPI ranks via decomposePar / mpirun / reconstructPar. The same N is written
into system/decomposeParDict at render time.
"""

import argparse
import json
import logging
import math
import shutil
import subprocess
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = PROJECT_ROOT / "cases"
TEMPLATES_ROOT = PROJECT_ROOT / "openfoam_template"
OPENFOAM_BASHRC = Path("/opt/openfoam12/etc/bashrc")

CHORD = 1.0
MESH_SPAN = 0.05
NU = 1.5e-5
RHO = 1.225
TURBULENCE_INTENSITY = 0.01
# Regime C (transitional low-Re) needs a clean-tunnel free-stream so the
# laminar BL survives to form the separation bubble before transition; a 1%
# free-stream would force premature bypass transition. CLAUDE.md §5 / §10.
TURBULENCE_INTENSITY_BY_REGIME = {"C": 0.001}
TURBULENCE_LENGTH_SCALE = 0.07 * CHORD
CMU = 0.09
DEFAULT_JOBS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the OpenFOAM base case into cases/case_XXXX and optionally "
            "run foamRun for cases that already contain constant/polyMesh."
        )
    )
    parser.add_argument(
        "--case-id",
        type=str,
        nargs="*",
        help=(
            "Specific case IDs or ranges. "
            "Examples: 1 3 5, 1-10, 1-5 8 10-12. "
            "Defaults to all cases with params.json."
        ),
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run foamRun after rendering for cases that already have a mesh.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help=(
            "Number of MPI subdomains written to system/decomposeParDict. "
            "When --run is enabled and --jobs > 1, each case is solved in "
            "parallel on this many cores (cases still run one-at-a-time)."
        ),
    )
    return parser.parse_args()


def fmt(value: float) -> str:
    return f"{value:.10g}"

def expand_case_ids(case_args: list[str] | None) -> list[int] | None:
    """
    Expand command-line case IDs.

    Examples
    --------
    ["1", "3", "5"]
        -> [1, 3, 5]

    ["1-5"]
        -> [1, 2, 3, 4, 5]

    ["1-5", "8", "10-12"]
        -> [1, 2, 3, 4, 5, 8, 10, 11, 12]
    """
    if not case_args:
        return None

    case_ids: set[int] = set()

    for item in case_args:
        if "-" in item:
            try:
                start_str, end_str = item.split("-", 1)
                start = int(start_str)
                end = int(end_str)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid case range '{item}'. Expected format like '1-10'."
                ) from exc

            if start > end:
                raise argparse.ArgumentTypeError(
                    f"Invalid case range '{item}': start must be <= end."
                )

            case_ids.update(range(start, end + 1))
        else:
            try:
                case_ids.add(int(item))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid case ID '{item}'."
                ) from exc

    return sorted(case_ids)

def collect_case_dirs(case_ids: list[int] | None) -> list[Path]:
    if case_ids:
        case_dirs = [CASES_DIR / f"case_{case_id:04d}" for case_id in case_ids]
    else:
        case_dirs = sorted(
            case_dir
            for case_dir in CASES_DIR.glob("case_*")
            if (case_dir / "params.json").exists()
        )

    missing = [case_dir for case_dir in case_dirs if not (case_dir / "params.json").exists()]
    if missing:
        missing_names = ", ".join(case_dir.name for case_dir in missing)
        raise FileNotFoundError(f"Missing params.json for: {missing_names}")

    return case_dirs


def load_params(case_dir: Path) -> dict[str, float | str]:
    payload = json.loads((case_dir / "params.json").read_text())
    regime = payload.get("regime")
    if regime is None:
        regime = "A"
        log.info("%s: params.json has no 'regime' field — defaulting to A", case_dir.name)
    return {
        "alpha_deg": float(payload["alpha_deg"]),
        "Re": float(payload["Re"]),
        "thickness": float(payload["thickness"]),
        "regime": str(regime),
    }


def build_render_context(
    alpha_deg: float, reynolds_number: float, nprocs: int, regime: str = "A"
) -> dict[str, str]:
    alpha_rad = math.radians(alpha_deg)
    u_inf = reynolds_number * NU / CHORD
    ux = u_inf * math.cos(alpha_rad)
    uy = u_inf * math.sin(alpha_rad)
    lift_x = -math.sin(alpha_rad)
    lift_y = math.cos(alpha_rad)
    drag_x = math.cos(alpha_rad)
    drag_y = math.sin(alpha_rad)

    turbulence_intensity = TURBULENCE_INTENSITY_BY_REGIME.get(regime, TURBULENCE_INTENSITY)
    k_inf = max(1.5 * (u_inf * turbulence_intensity) ** 2, 1e-10)
    omega_inf = max(
        math.sqrt(k_inf) / (CMU ** 0.25 * TURBULENCE_LENGTH_SCALE),
        1e-6,
    )
    # Freestream eddy viscosity, used to SEED the nut internalField. Starting
    # from nut=0 leaves the first SIMPLE iterations effectively laminar, which
    # lets the near-stall LE boundary layer separate before the turbulence
    # field develops — the solver then settles on the spurious fully-separated
    # branch (Cl ~0.26 independent of alpha in B validation). Seeding nut with
    # the freestream turbulent value keeps the BL energised through startup.
    nut_inf = k_inf / omega_inf

    return {
        "UX": fmt(ux),
        "UY": fmt(uy),
        "UINF": fmt(u_inf),
        "LIFTDIR_X": fmt(lift_x),
        "LIFTDIR_Y": fmt(lift_y),
        "DRAGDIR_X": fmt(drag_x),
        "DRAGDIR_Y": fmt(drag_y),
        "AREF": fmt(CHORD * MESH_SPAN),
        "KINF": fmt(k_inf),
        "OMEGAINF": fmt(omega_inf),
        "NUTINF": fmt(nut_inf),
        "RHO": fmt(RHO),
        "NPROCS": str(nprocs),
    }


def copy_static_template_files(case_dir: Path, template_dir: Path) -> None:
    for source_path in template_dir.rglob("*"):
        relative_path = source_path.relative_to(template_dir)
        target_path = case_dir / relative_path

        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue

        if source_path.name.endswith(".template"):
            continue

        if source_path.suffix == ".md":
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def render_template_files(case_dir: Path, context: dict[str, str], template_dir: Path) -> None:
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    for source_path in template_dir.rglob("*.template"):
        template_name = str(source_path.relative_to(template_dir))
        target_relative = template_name.removesuffix(".template")
        target_path = case_dir / target_relative
        target_path.parent.mkdir(parents=True, exist_ok=True)

        rendered = env.get_template(template_name).render(**context)
        target_path.write_text(rendered.rstrip() + "\n")


def render_case(case_dir: Path, nprocs: int) -> None:
    params = load_params(case_dir)
    regime = params["regime"]
    template_dir = TEMPLATES_ROOT / f"regime_{regime}"
    if not template_dir.exists():
        raise FileNotFoundError(
            f"No template for regime '{regime}': {template_dir} does not exist"
        )

    context = build_render_context(params["alpha_deg"], params["Re"], nprocs, regime)
    copy_static_template_files(case_dir, template_dir)
    render_template_files(case_dir, context, template_dir)

    log.info(
        "Rendered %s (regime %s)  alpha=%5.2f deg  Re=%.3e  t=%.4f  Uinf=%s m/s",
        case_dir.name,
        regime,
        params["alpha_deg"],
        params["Re"],
        params["thickness"],
        context["UINF"],
    )


def has_mesh(case_dir: Path) -> bool:
    return (case_dir / "constant" / "polyMesh" / "boundary").exists()


def run_case(case_dir: Path, nprocs: int) -> None:
    # Wall-resolved kOmegaSST diverges from a uniform U field on AR~10^3
    # boundary-layer cells, so initialise U/p with potentialFlow first.
    if nprocs > 1:
        command = (
            f"source {OPENFOAM_BASHRC} && "
            f"decomposePar -force > log.decomposePar 2>&1 && "
            f"mpirun --oversubscribe -np {nprocs} potentialFoam -initialiseUBCs -parallel "
            f"> log.potentialFoam 2>&1 && "
            f"mpirun --oversubscribe -np {nprocs} foamRun -parallel "
            f"> log.simpleFoam 2>&1 && "
            f"reconstructPar -latestTime > log.reconstructPar 2>&1"
        )
    else:
        command = (
            f"source {OPENFOAM_BASHRC} && "
            f"potentialFoam -initialiseUBCs > log.potentialFoam 2>&1 && "
            f"foamRun > log.simpleFoam 2>&1"
        )

    result = subprocess.run(
        ["bash", "-lc", command],
        cwd=case_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error("foamRun failed in %s: %s", case_dir, result.stderr[-500:])
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)

    log.info("Completed foamRun for %s (np=%d)", case_dir.name, nprocs)


def run_cases(case_dirs: list[Path], nprocs: int) -> None:
    for case_dir in case_dirs:
        run_case(case_dir, nprocs)


def main() -> None:
    args = parse_args()

    if not TEMPLATES_ROOT.exists():
        raise FileNotFoundError(f"{TEMPLATES_ROOT} not found")
    if not OPENFOAM_BASHRC.exists():
        raise FileNotFoundError(f"{OPENFOAM_BASHRC} not found")

    case_ids = expand_case_ids(args.case_id)
    case_dirs = collect_case_dirs(case_ids)
    if not case_dirs:
        log.warning("No case directories with params.json were found under %s", CASES_DIR)
        return

    nprocs = max(args.jobs, 1)
    for case_dir in case_dirs:
        render_case(case_dir, nprocs)

    if not args.run:
        return

    runnable_cases: list[Path] = []
    for case_dir in case_dirs:
        if has_mesh(case_dir):
            runnable_cases.append(case_dir)
        else:
            log.warning("Skipping %s for execution: constant/polyMesh not found", case_dir.name)

    if not runnable_cases:
        log.warning("No rendered cases have a mesh yet; nothing was executed")
        return

    run_cases(runnable_cases, nprocs)


if __name__ == "__main__":
    main()

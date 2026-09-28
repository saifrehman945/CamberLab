"""
Module: scripts/airfrans/io.py
Purpose: Read the locally downloaded PLAID-datasets/AirfRANS_remeshed dataset
         (Hugging Face parquet shards of pickled PLAID samples) and its card.

The data is never downloaded from here. The user fetches it once with

    huggingface-cli download PLAID-datasets/AirfRANS_remeshed \
        --repo-type dataset --local-dir data/airfrans_remeshed

and every script resolves the location as --data-dir > $AIRFRANS_DIR >
data/airfrans_remeshed.
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "airfrans_remeshed"
EXPECTED_SHARDS = 3
EXPECTED_SAMPLES = 1000

# Scalar names as published in the dataset card (in_/out_scalars_names).
SCALAR_AOA = "angle_of_attack"       # radians (verified in Phase 2)
SCALAR_UINF = "inlet_velocity"       # m/s
SCALAR_CD = "C_D"
SCALAR_CL = "C_L"

# AirfRANS generator constants: chord 1 m, T = 298.15 K.
CHORD = 1.0
TEMPERATURE = 298.15


def nu_air(T: float = TEMPERATURE) -> float:
    """Kinematic viscosity of air (m^2/s), the polynomial used by AirfRANS' generator."""
    return -3.400747e-6 + 3.452139e-8 * T + 1.00881778e-10 * T**2 - 1.363528e-14 * T**3


NU = nu_air()


def reynolds(U_inf: np.ndarray | float) -> np.ndarray | float:
    return np.asarray(U_inf) * CHORD / NU if np.ndim(U_inf) else U_inf * CHORD / NU


def resolve_data_dir(cli_value: str | Path | None = None) -> Path:
    """--data-dir > $AIRFRANS_DIR > data/airfrans_remeshed; fail loudly if incomplete."""
    if cli_value:
        data_dir = Path(cli_value)
    elif os.environ.get("AIRFRANS_DIR"):
        data_dir = Path(os.environ["AIRFRANS_DIR"])
    else:
        data_dir = DEFAULT_DATA_DIR
    data_dir = data_dir.resolve()
    shards = shard_paths(data_dir)
    if len(shards) != EXPECTED_SHARDS:
        raise FileNotFoundError(
            f"Expected {EXPECTED_SHARDS} parquet shards under {data_dir}/data, found "
            f"{len(shards)}. Download the dataset first (see scripts/airfrans/io.py)."
        )
    return data_dir


def shard_paths(data_dir: Path) -> list[Path]:
    return sorted((data_dir / "data").glob("all_samples-*.parquet"))


def load_card(data_dir: Path) -> dict:
    """Parse the YAML front matter of the local dataset card (README.md)."""
    text = (data_dir / "README.md").read_text()
    front = text.split("---", 2)[1]
    return yaml.safe_load(front)


def load_splits(data_dir: Path) -> dict[str, np.ndarray]:
    """Official AirfRANS task splits, as row indices into `all_samples`."""
    raw = load_card(data_dir)["dataset_info"]["description"]["split"]
    return {name: np.asarray(ids, dtype=np.int32) for name, ids in raw.items()}


class RawSampleReader:
    """Random access to the pickled sample bytes across all shards, in row order."""

    def __init__(self, data_dir: Path):
        self._files = [pq.ParquetFile(p) for p in shard_paths(data_dir)]
        # (file index, row group index, first global row of the group, n rows)
        self._groups: list[tuple[int, int, int, int]] = []
        start = 0
        for fi, pf in enumerate(self._files):
            for gi in range(pf.metadata.num_row_groups):
                n = pf.metadata.row_group(gi).num_rows
                self._groups.append((fi, gi, start, n))
                start += n
        self.n_samples = start
        self._cache_key: tuple[int, int] | None = None
        self._cache_col = None

    def __len__(self) -> int:
        return self.n_samples

    def raw_bytes(self, i: int) -> bytes:
        if not 0 <= i < self.n_samples:
            raise IndexError(i)
        for fi, gi, start, n in self._groups:
            if start <= i < start + n:
                if self._cache_key != (fi, gi):
                    table = self._files[fi].read_row_group(gi, columns=["sample"])
                    self._cache_col = table.column("sample")
                    self._cache_key = (fi, gi)
                return self._cache_col[i - start].as_py()
        raise IndexError(i)


def deserialise(raw: bytes):
    """Bytes -> plaid Sample, exactly as the dataset card prescribes.

    This is the one sanctioned use of pickle in the project (CLAUDE.md otherwise
    forbids it): the dataset is distributed as pickled PLAID sample dicts, and
    the card's loader is `Sample.model_validate(pickle.loads(...))`. Only
    pyplaid==0.1.7 accepts this serialised schema (pinned in requirements.txt).
    """
    from plaid.containers.sample import Sample

    return Sample.model_validate(pickle.loads(raw))


@dataclass
class SampleData:
    """The parts of one AirfRANS sample a coefficient surrogate needs."""

    index: int
    scalars: dict[str, float]
    x: np.ndarray                 # node coordinates
    y: np.ndarray
    triangles: np.ndarray         # (n_tri, 3) zero-based node indices
    implicit_distance: np.ndarray  # per-node signed distance to the aerofoil


def extract(sample, index: int) -> SampleData:
    """Pull scalars, nodes, triangles and wall distance out of a plaid Sample."""
    scalars = {str(k): float(sample.get_scalar(k)) for k in sample.get_scalar_names()}
    nodes = np.asarray(sample.get_nodes(), dtype=np.float64)
    elements = sample.get_elements()
    tri_key = next(k for k in elements if "TRI" in str(k).upper())
    triangles = np.asarray(elements[tri_key], dtype=np.int64).reshape(-1, 3)
    dist = np.asarray(sample.get_field("implicit_distance"), dtype=np.float64)
    return SampleData(index=index, scalars=scalars, x=nodes[:, 0], y=nodes[:, 1],
                      triangles=triangles, implicit_distance=dist)


def iter_samples(data_dir: Path, indices=None):
    """Yield SampleData for the given row indices (default: all, in order)."""
    reader = RawSampleReader(data_dir)
    for i in (range(len(reader)) if indices is None else indices):
        yield extract(deserialise(reader.raw_bytes(int(i))), int(i))

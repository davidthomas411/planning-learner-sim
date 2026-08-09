"""Deterministic case-level splits created before trajectory generation."""

from dataclasses import dataclass
import csv
import hashlib
import io

import numpy as np


@dataclass(frozen=True)
class SplitConfig:
    train_cases: int = 7000
    validation_cases: int = 1000
    iid_test_cases: int = 1000
    ood_test_cases: int = 1000
    primary_seed_start: int = 100000
    ood_seed_start: int = 900000
    shuffle_seed: int = 20260809
    split_schema_version: str = "1.0"


@dataclass(frozen=True)
class SplitRow:
    case_id: str
    seed: int
    split: str
    split_ordinal: int
    generator_condition: str
    split_schema_version: str


def build_split_rows(config: SplitConfig | None = None) -> tuple[SplitRow, ...]:
    """Assign anatomy seeds once, before endpoints or trajectories exist."""

    cfg = config or SplitConfig()
    counts = (cfg.train_cases, cfg.validation_cases, cfg.iid_test_cases, cfg.ood_test_cases)
    if any(value < 1 for value in counts):
        raise ValueError("Every split must contain at least one case")
    primary_count = cfg.train_cases + cfg.validation_cases + cfg.iid_test_cases
    primary_seeds = np.arange(
        cfg.primary_seed_start,
        cfg.primary_seed_start + primary_count,
        dtype=np.int64,
    )
    rng = np.random.default_rng(cfg.shuffle_seed)
    rng.shuffle(primary_seeds)
    boundaries = (cfg.train_cases, cfg.train_cases + cfg.validation_cases)
    seed_groups = (
        ("train", primary_seeds[: boundaries[0]], "primary"),
        ("validation", primary_seeds[boundaries[0] : boundaries[1]], "primary"),
        ("iid_test", primary_seeds[boundaries[1] :], "primary"),
        (
            "ood_test",
            np.arange(cfg.ood_seed_start, cfg.ood_seed_start + cfg.ood_test_cases, dtype=np.int64),
            "ood_reserved",
        ),
    )
    rows: list[SplitRow] = []
    for split, seeds, condition in seed_groups:
        for ordinal, seed_value in enumerate(seeds):
            seed = int(seed_value)
            prefix = "synthetic3d-ood" if split == "ood_test" else "synthetic3d"
            rows.append(
                SplitRow(
                    case_id=f"{prefix}-{seed:06d}",
                    seed=seed,
                    split=split,
                    split_ordinal=ordinal,
                    generator_condition=condition,
                    split_schema_version=cfg.split_schema_version,
                )
            )
    case_ids = [row.case_id for row in rows]
    seeds = [row.seed for row in rows]
    if len(case_ids) != len(set(case_ids)) or len(seeds) != len(set(seeds)):
        raise RuntimeError("Split construction produced duplicate cases")
    return tuple(rows)


def render_split_manifest(rows: tuple[SplitRow, ...]) -> str:
    """Return canonical UTF-8 CSV content suitable for hashing."""

    handle = io.StringIO(newline="")
    fieldnames = [
        "case_id",
        "seed",
        "split",
        "split_ordinal",
        "generator_condition",
        "split_schema_version",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: getattr(row, name) for name in fieldnames})
    return handle.getvalue()


def split_manifest_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

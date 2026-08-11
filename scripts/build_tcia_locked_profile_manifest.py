"""Assign one fixed starting-plan profile to each locked TCIA patient."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import OrderedDict
from pathlib import Path


PROFILE_QUOTAS = OrderedDict(
    (
        ("balanced_reference", 10),
        ("oar_guarded", 21),
        ("hotspot_stress", 35),
        ("conformity_stress", 35),
    )
)


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def proportional_quotas(
    quotas: dict[str, int],
    stratum_size: int,
    total_size: int,
) -> dict[str, int]:
    """Allocate exact stratum counts by largest proportional remainder."""

    raw = {
        profile: count * stratum_size / total_size
        for profile, count in quotas.items()
    }
    allocated = {profile: int(value) for profile, value in raw.items()}
    remaining = stratum_size - sum(allocated.values())
    order = {profile: index for index, profile in enumerate(quotas)}
    ranked = sorted(
        quotas,
        key=lambda profile: (-(raw[profile] - allocated[profile]), order[profile]),
    )
    for profile in ranked[:remaining]:
        allocated[profile] += 1
    return allocated


def spread_profiles(quotas: dict[str, int]) -> list[str]:
    """Spread fixed profile counts over an ordered anatomy stratum."""

    total = sum(quotas.values())
    assigned = {profile: 0 for profile in quotas}
    sequence: list[str] = []
    order = {profile: index for index, profile in enumerate(quotas)}
    for position in range(1, total + 1):
        eligible = [
            profile
            for profile, quota in quotas.items()
            if assigned[profile] < quota
        ]
        selected = max(
            eligible,
            key=lambda profile: (
                position * quotas[profile] / total - assigned[profile],
                -order[profile],
            ),
        )
        sequence.append(selected)
        assigned[selected] += 1
    return sequence


def build_locked_manifest(
    records: list[dict[str, str]],
    development_ids: set[str],
    quotas: dict[str, int] | None = None,
) -> list[dict[str, str | int | float]]:
    """Exclude development patients and assign one profile per remaining patient."""

    profile_quotas = OrderedDict(quotas or PROFILE_QUOTAS)
    expected_count = sum(profile_quotas.values())
    patient_ids = [row["patient_id"] for row in records]
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("The anatomy metrics contain duplicate patient identifiers")
    missing_development = development_ids - set(patient_ids)
    if missing_development:
        raise ValueError(
            f"Development patients are missing from the anatomy metrics: {sorted(missing_development)}"
        )
    locked = [row for row in records if row["patient_id"] not in development_ids]
    if len(locked) != expected_count:
        raise ValueError(
            f"The profile quotas require {expected_count} patients, but {len(locked)} remain"
        )

    strata = {
        "margin_only": [
            row for row in locked if parse_bool(row["margin_only_primary_eligible"])
        ],
        "interface_overlap": [
            row for row in locked if not parse_bool(row["margin_only_primary_eligible"])
        ],
    }
    margin_quotas = proportional_quotas(
        profile_quotas,
        len(strata["margin_only"]),
        expected_count,
    )
    stratum_quotas = {
        "margin_only": margin_quotas,
        "interface_overlap": {
            profile: profile_quotas[profile] - margin_quotas[profile]
            for profile in profile_quotas
        },
    }

    assignments: list[dict[str, str | int | float]] = []
    for stratum_name, stratum_records in strata.items():
        ordered_records = sorted(
            stratum_records,
            key=lambda row: (
                float(row["maximum_ptv_oar_overlap_fraction"]),
                row["patient_id"],
            ),
        )
        profiles = spread_profiles(stratum_quotas[stratum_name])
        for rank, (row, profile) in enumerate(
            zip(ordered_records, profiles, strict=True),
            start=1,
        ):
            assignments.append(
                {
                    "patient_id": row["patient_id"],
                    "case_id": row["case_id"],
                    "anatomy_stratum": stratum_name,
                    "stratum_overlap_rank": rank,
                    "stratum_size": len(ordered_records),
                    "maximum_ptv_oar_overlap_fraction": float(
                        row["maximum_ptv_oar_overlap_fraction"]
                    ),
                    "starting_profile": profile,
                    "assignment_rule": (
                        "development15_excluded; proportional_profile_quota_by_stratum; "
                        "deficit_spread_across_maximum_ptv_oar_overlap_rank"
                    ),
                }
            )

    assignments.sort(key=lambda row: str(row["patient_id"]))
    for index, row in enumerate(assignments, start=1):
        row["assignment_order"] = index
    counts = {
        profile: sum(row["starting_profile"] == profile for row in assignments)
        for profile in profile_quotas
    }
    if counts != dict(profile_quotas):
        raise RuntimeError(f"Profile assignment counts do not match quotas: {counts}")
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the locked 101-patient TCIA profile assignment"
    )
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--development-subject-file", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()

    with args.metrics_csv.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    development_ids = {
        value.strip()
        for value in args.development_subject_file.read_text(encoding="utf-8").splitlines()
        if value.strip()
    }
    assignments = build_locked_manifest(records, development_ids)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "assignment_order",
        "patient_id",
        "case_id",
        "anatomy_stratum",
        "stratum_overlap_rank",
        "stratum_size",
        "maximum_ptv_oar_overlap_fraction",
        "starting_profile",
        "assignment_rule",
    ]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(assignments)
    manifest_sha256 = hashlib.sha256(args.output_csv.read_bytes()).hexdigest()

    summary = {
        "status": "locked",
        "patients": len(assignments),
        "development_patients_excluded": len(development_ids),
        "profile_counts": {
            profile: sum(row["starting_profile"] == profile for row in assignments)
            for profile in PROFILE_QUOTAS
        },
        "anatomy_strata": {
            stratum: sum(row["anatomy_stratum"] == stratum for row in assignments)
            for stratum in ("margin_only", "interface_overlap")
        },
        "manifest_sha256": manifest_sha256,
        "metrics_csv": args.metrics_csv.as_posix(),
        "development_subject_file": args.development_subject_file.as_posix(),
        "output_csv": args.output_csv.as_posix(),
    }
    summary_path = args.summary_json or args.output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

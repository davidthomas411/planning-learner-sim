from collections import Counter

from dosim_sim.splits import (
    SplitConfig,
    build_split_rows,
    render_split_manifest,
    split_manifest_sha256,
)


def test_split_counts_and_case_ids_are_disjoint() -> None:
    config = SplitConfig(train_cases=12, validation_cases=3, iid_test_cases=4, ood_test_cases=5)
    rows = build_split_rows(config)
    assert Counter(row.split for row in rows) == {
        "train": 12,
        "validation": 3,
        "iid_test": 4,
        "ood_test": 5,
    }
    assert len({row.case_id for row in rows}) == len(rows)
    assert len({row.seed for row in rows}) == len(rows)


def test_split_manifest_is_reproducible_and_sensitive_to_seed() -> None:
    first = render_split_manifest(build_split_rows(SplitConfig(shuffle_seed=4)))
    second = render_split_manifest(build_split_rows(SplitConfig(shuffle_seed=4)))
    changed = render_split_manifest(build_split_rows(SplitConfig(shuffle_seed=5)))
    assert first == second
    assert split_manifest_sha256(first) == split_manifest_sha256(second)
    assert first != changed


def test_split_manifest_contains_no_endpoint_or_trajectory_fields() -> None:
    content = render_split_manifest(
        build_split_rows(SplitConfig(train_cases=2, validation_cases=1, iid_test_cases=1, ood_test_cases=1))
    )
    header = content.splitlines()[0].split(",")
    assert "action" not in header
    assert "dose" not in header
    assert "final_state" not in header

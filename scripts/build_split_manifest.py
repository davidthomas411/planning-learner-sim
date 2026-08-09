import argparse
import json
from collections import Counter
from pathlib import Path

from dosim_sim.splits import (
    SplitConfig,
    build_split_rows,
    render_split_manifest,
    split_manifest_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze case-level splits before trajectory generation")
    parser.add_argument("--output", type=Path, default=Path("outputs/splits/case_split_manifest.csv"))
    parser.add_argument("--train", type=int, default=7000)
    parser.add_argument("--validation", type=int, default=1000)
    parser.add_argument("--iid-test", type=int, default=1000)
    parser.add_argument("--ood-test", type=int, default=1000)
    parser.add_argument("--shuffle-seed", type=int, default=20260809)
    args = parser.parse_args()
    cfg = SplitConfig(
        train_cases=args.train,
        validation_cases=args.validation,
        iid_test_cases=args.iid_test,
        ood_test_cases=args.ood_test,
        shuffle_seed=args.shuffle_seed,
    )
    rows = build_split_rows(cfg)
    content = render_split_manifest(rows)
    digest = split_manifest_sha256(content)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content.encode("utf-8"))
    digest_path = args.output.with_suffix(args.output.suffix + ".sha256")
    digest_path.write_text(f"{digest}  {args.output.name}\n", encoding="ascii")
    print(
        json.dumps(
            {
                "path": str(args.output),
                "sha256": digest,
                "counts": dict(Counter(row.split for row in rows)),
                "total_cases": len(rows),
                "trajectory_fields_present": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

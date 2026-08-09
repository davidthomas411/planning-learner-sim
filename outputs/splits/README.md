# Case split manifest

`case_split_manifest.csv` assigns 10,000 unique anatomy seeds before any endpoint, dose, action, or trajectory is generated:

- 7,000 training cases;
- 1,000 validation cases;
- 1,000 IID test cases;
- 1,000 reserved OOD test cases.

The primary seeds are shuffled deterministically with seed `20260809`. OOD rows are labeled `ood_reserved`; the exact OOD generator condition must be calibrated and frozen before those anatomies are generated. The manifest contains no outcome or trajectory fields, preventing plan quality from influencing the split.

The companion `.sha256` file protects the exact case assignment. Any change to ordering, IDs, seeds, or split labels changes the digest.

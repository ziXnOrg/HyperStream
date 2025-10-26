# CI Baseline Refresh (with Provenance)

This guide describes how to regenerate performance baselines for the CI Perf Regression (NDJSON) checks and capture full provenance for reproducibility.

## When to Refresh
- OS image or toolchain upgrades (e.g., ubuntu-24.04 / GCC 13.x)
- Significant performance work (kernels, memory layout, algorithmic changes)
- Runner hardware/image drift suspected (broad regressions with no code change)

## What Gets Produced
- `ci/perf_baseline/am_baseline_${os}_${timestamp}.ndjson`
- `ci/perf_baseline/cluster_baseline_${os}_${timestamp}.ndjson`
- `ci/perf_baseline/provenance_${os}_${timestamp}.json`
- Convenience pointers:
  - `ci/perf_baseline/linux_baseline.ndjson` (am+cluster concatenated)
  - `ci/perf_baseline/linux_baseline_provenance.json`

## How to Refresh (Ubuntu runner)

1) Use the provided script on a clean runner (ubuntu-latest):

```bash
bash scripts/refresh_baselines.sh
```

2) Verify integrity
- Run the script twice; ensure < 5% variance between runs in the same environment
- Spot-check aggregates and outliers (qps, gbps)

3) Commit and Review
- Commit the new NDJSON baselines and provenance JSON
- PR description must include:
  - OS image/version, compiler, cmake version
  - SHA/branch, timestamp
  - CI run links for both generation runs
- Require 2 approvals for baseline changes

## Provenance Fields
The provenance JSON includes:
- `timestamp_utc` (ISO 8601)
- `git_sha`, `git_branch`
- `runner_os`, `image_os`, `image_version`
- `uname`, `/etc/os-release` contents
- `cmake_version`, `compiler_version`
- `cmake_cache` (build type, flags)
- `artifacts` (baseline filenames)

## Notes
- Keep baselines per-OS; do not reuse across different images/toolchains.
- If macOS/Windows baselines need refresh, follow the same pattern and adjust `OS_TAG`.
- Do not widen tolerances in CI unless explicitly approved; prefer re-baselining with provenance.


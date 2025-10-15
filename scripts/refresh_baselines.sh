#!/usr/bin/env bash
set -euo pipefail

# Refresh performance baselines with provenance (intended for ubuntu-latest runners)
# Outputs:
# - ci/perf_baseline/am_baseline_${OS_TAG}_${DATE}.ndjson
# - ci/perf_baseline/cluster_baseline_${OS_TAG}_${DATE}.ndjson
# - ci/perf_baseline/provenance_${OS_TAG}_${DATE}.json
# Also updates convenience pointers:
# - ci/perf_baseline/linux_baseline.ndjson (concatenated am+cluster)
# - ci/perf_baseline/linux_baseline_provenance.json

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

OS_TAG=${OS_TAG:-linux}
DATE_UTC=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR="ci/perf_baseline"
mkdir -p "$OUT_DIR"

# Detect environment metadata
RUNNER_OS_VAL=${RUNNER_OS:-}
IMAGE_OS_VAL=${ImageOS:-}
IMAGE_VER_VAL=${ImageVersion:-}
GIT_SHA=$(git rev-parse HEAD)
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
CXX_VER=$( (c++ --version || g++ --version || clang++ --version) 2>/dev/null | head -n1 || echo "unknown")
CMAKE_VER=$(cmake --version | head -n1 || echo "unknown")
UNAME_A=$(uname -a || echo "unknown")
OS_RELEASE=$( (cat /etc/os-release 2>/dev/null || true) | tr -d '\r' )

# Build (Release)
BUILD_DIR="build_perf"
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --config Release --target am_bench cluster_bench

# Run benchmarks (NDJSON mode)
AM_OUT="$OUT_DIR/am_baseline_${OS_TAG}_${DATE_UTC}.ndjson"
CL_OUT="$OUT_DIR/cluster_baseline_${OS_TAG}_${DATE_UTC}.ndjson"

"$BUILD_DIR/benchmarks/am_bench" --json --warmup_ms=100 --measure_ms=200 --samples=5 | tee "$AM_OUT" >/dev/null
"$BUILD_DIR/benchmarks/cluster_bench" --json --warmup_ms=100 --measure_ms=200 --samples=5 | tee "$CL_OUT" >/dev/null

# Combine for convenience pointer
cat "$AM_OUT" "$CL_OUT" > "$OUT_DIR/linux_baseline.ndjson"

# Write provenance
PROV_JSON="$OUT_DIR/provenance_${OS_TAG}_${DATE_UTC}.json"
cat > "$PROV_JSON" <<EOF
{
  "timestamp_utc": "$DATE_UTC",
  "git_sha": "$GIT_SHA",
  "git_branch": "$BRANCH_NAME",
  "runner_os": "${RUNNER_OS_VAL}",
  "image_os": "${IMAGE_OS_VAL}",
  "image_version": "${IMAGE_VER_VAL}",
  "uname": "${UNAME_A}",
  "os_release": "$(echo "$OS_RELEASE" | sed 's/"/\\"/g')",
  "cmake_version": "${CMAKE_VER}",
  "compiler_version": "${CXX_VER}",
  "cmake_cache": {
    "build_type": "Release"
  },
  "artifacts": {
    "am_baseline": "$(basename "$AM_OUT")",
    "cluster_baseline": "$(basename "$CL_OUT")",
    "combined": "linux_baseline.ndjson"
  }
}
EOF

cp "$PROV_JSON" "$OUT_DIR/linux_baseline_provenance.json"

echo "Baselines written to $OUT_DIR"


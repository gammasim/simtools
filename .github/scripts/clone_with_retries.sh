#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 SOURCE_URL REF TARGET [EXPECTED_REVISION]" >&2
  exit 2
fi

source_url=$1
source_ref=$2
target=$3
expected_revision=${4:-}

for attempt in 1 2 3 4 5; do
  rm -rf -- "$target"
  if git -c http.connectTimeout=30 \
    -c http.lowSpeedLimit=1000 \
    -c http.lowSpeedTime=60 \
    clone --depth 1 --branch "$source_ref" "$source_url" "$target"; then
    actual_revision=$(git -C "$target" rev-parse HEAD)
    if [[ -n "$expected_revision" && "$actual_revision" != "$expected_revision" ]]; then
      echo "Revision mismatch for $target: expected $expected_revision, got $actual_revision" >&2
      exit 1
    fi
    printf '%s\n' "$actual_revision"
    exit 0
  fi

  if [[ "$attempt" -eq 5 ]]; then
    echo "Failed to clone $source_ref into $target after $attempt attempts" >&2
    exit 1
  fi
  sleep $((attempt * 15))
done

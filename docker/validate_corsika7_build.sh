#!/usr/bin/env bash

# Validate each CORSIKA binary against its generated configuration and startup banner.
# This prevents incomplete, stale, or incorrectly configured builds from entering the runtime image.

set -euo pipefail

if [[ $# -ne 7 ]]; then
    echo "Usage: $0 CONFIG EXECUTABLE VERSION MODEL GEOMETRY OPTIMIZATION REPORT_DIR" >&2
    exit 2
fi

config_file=$1
executable=$2
corsika_version=$3
model=$4
geometry=$5
optimization=$6
report_dir=$7

require_define() {
    local symbol=$1
    if ! grep -Eq "^#define[[:space:]]+${symbol}([[:space:]]|$)" "$config_file"; then
        echo "Required macro ${symbol} is not defined in ${config_file}." >&2
        exit 1
    fi
    return 0
}

reject_define() {
    local symbol=$1
    if grep -Eq "^#define[[:space:]]+${symbol}([[:space:]]|$)" "$config_file"; then
        echo "Unexpected macro ${symbol} is defined in ${config_file}." >&2
        exit 1
    fi
    return 0
}

[[ -x "$executable" ]] || {
    echo "CORSIKA executable is missing or not executable: ${executable}" >&2
    exit 1
}

require_define "__URQMD__"
if [[ "$model" == "qgs3" ]]; then
    require_define "__QGSJET__"
    require_define "__QGSIII__"
    require_define "__CACHE_QGSJETIII__"
    require_define "__CACHE_QGSJET_III__"
    reject_define "__EPOS__"
elif [[ "$model" == "epos" ]]; then
    require_define "__EPOS__"
    reject_define "__QGSJET__"
    reject_define "__QGSIII__"
else
    echo "Unsupported high-energy model: ${model}" >&2
    exit 2
fi

if [[ "$geometry" == "curved" ]]; then
    require_define "__CURVED__"
elif [[ "$geometry" == "flat" ]]; then
    reject_define "__CURVED__"
else
    echo "Unsupported atmosphere geometry: ${geometry}" >&2
    exit 2
fi

if [[ "$optimization" == "generic" ]]; then
    reject_define "__CERENKOPT__"
    reject_define "__VLIBM__"
else
    require_define "__CERENKOPT__"
    require_define "__VLIBM__"
fi

mkdir -p "$report_dir"
cp "$config_file" "${report_dir}/generated-config.h"
sha256sum "$executable" > "${report_dir}/executable.sha256"

if ! ldd "$executable" > "${report_dir}/linked-libraries.txt" 2>&1; then
    echo "Unable to inspect linked libraries for ${executable}." >&2
    exit 1
fi

strings "$executable" > "${report_dir}/executable.strings"
if [[ "$model" == "qgs3" ]]; then
    grep -q "QGSJET-III MODEL" "${report_dir}/executable.strings"
    grep -q "QUARK GLUON STRING JET - III MODEL" "${report_dir}/executable.strings"
else
    grep -q "EPOS MODEL" "${report_dir}/executable.strings"
fi

startup_validation="skipped (requires ${optimization} CPU support)"
startup_status="not run"
if [[ "$optimization" == "generic" ]]; then
    printf 'EXIT\n' > "${report_dir}/startup.input"
    set +e
    timeout 20 "$executable" \
        < "${report_dir}/startup.input" \
        > "${report_dir}/startup.log" 2>&1
    startup_status=$?
    set -e
    if [[ $startup_status -eq 124 || $startup_status -eq 126 || $startup_status -eq 127 || \
        $startup_status -ge 128 ]]; then
        echo "CORSIKA startup probe failed with status ${startup_status}." >&2
        exit 1
    fi

    display_version="7.${corsika_version:1}"
    display_version_pattern=${display_version//./\\.}
    grep -Eq "NUMBER OF VERSION[[:space:]]*:[[:space:]]*${display_version_pattern}" \
        "${report_dir}/startup.log"
    if grep -q "CORSIKA version is .* but IACT interface was adapted" \
        "${report_dir}/startup.log"; then
        echo "CORSIKA/IACT version mismatch found during startup validation." >&2
        exit 1
    fi

    if [[ "$model" == "qgs3" ]]; then
        grep -q "QGSJET-III MODEL" "${report_dir}/startup.log"
    else
        grep -q "EPOS MODEL" "${report_dir}/startup.log"
    fi
    startup_validation="passed"
fi

{
    echo "corsika_version: ${corsika_version}"
    echo "model: ${model}"
    echo "geometry: ${geometry}"
    echo "optimization: ${optimization}"
    echo "executable: $(basename "$executable")"
    echo "macro_validation: passed"
    echo "binary_model_validation: passed"
    echo "linked_library_validation: passed"
    echo "startup_validation: ${startup_validation}"
    echo "startup_exit_status: ${startup_status}"
} > "${report_dir}/validation.txt"

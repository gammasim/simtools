#!/usr/bin/env python3
"""Run selected integration tests and compare their generated outputs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml
from astropy.table import Table

IGNORED_SUFFIXES = {".log", ".log_gz"}
BINARY_SUFFIXES = {".zst", ".simtel", ".png", ".jpg", ".jpeg", ".pdf"}
NORMALIZATION_PATTERNS = (
    (
        re.compile(
            r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
            r"(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b"
        ),
        "<TIMESTAMP>",
    ),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "<DATE>"),
    (
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-8][0-9a-fA-F]{3}-"
            r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
        ),
        "<UUID>",
    ),
    (re.compile(r"\b(?:[0-9a-f]{40}|[0-9a-f]{64})\b", re.IGNORECASE), "<GIT_HASH>"),
    (
        re.compile(r"(?m)^(?:host|hostname|user|username)\s*[:=]\s*[^\r\n]+", re.IGNORECASE),
        "<HOST_OR_USER>",
    ),
    (
        re.compile(r"simtools[-_ ]version\s*[:=]\s*[^\s,;]+", re.IGNORECASE),
        "simtools-version=<VERSION>",
    ),
    (re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:\\|/)(?:[^\s\"']+)", re.IGNORECASE), "<ROOT>"),
)


def normalize_text(value):
    """Replace volatile values in text using the shared normalization rules."""
    for pattern, replacement in NORMALIZATION_PATTERNS:
        value = pattern.sub(replacement, value)
    return value


def normalize_value(value):
    """Normalize strings recursively while preserving structured values."""
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {normalize_value(key): normalize_value(item) for key, item in value.items()}
    return value


def _equal_values(base, head):
    """Compare nested parsed structures, allowing tiny numeric differences."""
    if isinstance(base, dict) and isinstance(head, dict):
        return base.keys() == head.keys() and all(
            _equal_values(base[key], head[key]) for key in base
        )
    if isinstance(base, list) and isinstance(head, list):
        return len(base) == len(head) and all(
            _equal_values(left, right) for left, right in zip(base, head)
        )
    if (
        isinstance(base, (int, float))
        and not isinstance(base, bool)
        and isinstance(head, (int, float))
        and not isinstance(head, bool)
    ):
        return bool(np.isclose(base, head, rtol=1e-12, atol=0, equal_nan=True))
    return base == head


def _read_table(path):
    """Read an ECSV or FITS table with Astropy."""
    return Table.read(path)


def _compare_tables(base_path, head_path):
    """Return table differences for column metadata, values, and metadata."""
    base, head = _read_table(base_path), _read_table(head_path)
    differences = []
    if base.colnames != head.colnames:
        differences.append(f"columns: {base.colnames!r} != {head.colnames!r}")
        return differences
    if len(base) != len(head):
        differences.append(f"rows: {len(base)} != {len(head)}")
    for name in base.colnames:
        left, right = base[name], head[name]
        if str(left.unit) != str(right.unit):
            differences.append(f"{name} units: {left.unit!s} != {right.unit!s}")
        if left.dtype.kind in "iufc" and right.dtype.kind in "iufc":
            equal = np.isclose(left, right, rtol=1e-12, atol=0, equal_nan=True)
        else:
            equal = np.asarray(left == right)
        for index in np.flatnonzero(~equal)[:20]:
            differences.append(f"{name}[{index}]: {left[index]!r} != {right[index]!r}")
    base_meta = normalize_value(dict(base.meta))
    head_meta = normalize_value(dict(head.meta))
    if not _equal_values(base_meta, head_meta):
        differences.append("table metadata differs")
    return differences[:20]


def _compare_hdf5_attributes(name, item, other):
    """Return differences for one HDF5 object's attributes."""
    differences = []
    if set(item.attrs) != set(other.attrs):
        differences.append(f"{name} attribute names differ")
    for key, value in item.attrs.items():
        left_value = _normalize_hdf5_value(value)
        right_value = _normalize_hdf5_value(other.attrs[key]) if key in other.attrs else None
        if key not in other.attrs or not _equal_values(left_value, right_value):
            differences.append(f"{name} attribute {key!r} differs")
    return differences


def _normalize_hdf5_value(value):
    """Normalize strings and JSON-encoded HDF5 metadata recursively."""
    if isinstance(value, np.ndarray):
        return [_normalize_hdf5_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _normalize_hdf5_value(value.item())
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            return normalize_value(json.loads(value))
        except json.JSONDecodeError:
            return normalize_text(value)
    return normalize_value(value)


def _compare_hdf5_dataset(name, item, other):
    """Return differences for one HDF5 dataset."""
    if not isinstance(other, h5py.Dataset):
        return [f"{name}: dataset type differs"]
    left, right = item[()], other[()]
    if left.shape != right.shape:
        differences = [f"{name}: shape {left.shape} != {right.shape}"]
    elif left.dtype.kind in "iufc" and right.dtype.kind in "iufc":
        equal = np.all(np.isclose(left, right, rtol=1e-12, atol=0, equal_nan=True))
        differences = [] if equal else [f"{name}: numeric values differ"]
    else:
        equal = _equal_values(_normalize_hdf5_value(left), _normalize_hdf5_value(right))
        differences = [] if equal else [f"{name}: values differ"]
    return differences


def _hdf5_differences(base_path, head_path):
    """Compare HDF5 datasets and attributes, applying numeric tolerance."""
    differences = []
    with h5py.File(base_path, "r") as base, h5py.File(head_path, "r") as head:

        def visit(name, item):
            if name not in head:
                differences.append(f"missing dataset/group: {name}")
            else:
                differences.extend(_compare_hdf5_attributes(name, item, head[name]))
                if isinstance(item, h5py.Dataset):
                    differences.extend(_compare_hdf5_dataset(name, item, head[name]))

        base.visititems(visit)

        def check_extra(name, _item):
            if name not in base:
                differences.append(f"extra dataset/group: {name}")

        head.visititems(check_extra)
    return differences[:20]


def _compare_file(base_path, head_path):
    """Return differences for a pair of supported files."""
    suffix = base_path.suffix.lower()
    if suffix in {".ecsv", ".fits", ".fit"}:
        return _compare_tables(base_path, head_path)
    if suffix in {".h5", ".hdf5"}:
        return _hdf5_differences(base_path, head_path)
    if suffix == ".json":
        base, head = (
            normalize_value(json.loads(base_path.read_text())),
            normalize_value(json.loads(head_path.read_text())),
        )
        return [] if _equal_values(base, head) else ["JSON structures differ"]
    if suffix in {".yaml", ".yml"}:
        base, head = (
            normalize_value(yaml.safe_load(base_path.read_text())),
            normalize_value(yaml.safe_load(head_path.read_text())),
        )
        return [] if _equal_values(base, head) else ["YAML structures differ"]
    base_lines = normalize_text(base_path.read_text(errors="replace")).splitlines()
    head_lines = normalize_text(head_path.read_text(errors="replace")).splitlines()
    return [
        f"line {index + 1}: {left!r} != {right!r}"
        for index, (left, right) in enumerate(zip(base_lines, head_lines))
        if left != right
    ][:20] + (
        [f"line count: {len(base_lines)} != {len(head_lines)}"]
        if len(base_lines) != len(head_lines)
        else []
    )


def _relative_files(root):
    """Map comparable output paths to files, excluding pytest scaffolding."""
    files = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() in IGNORED_SUFFIXES:
            continue
        parts = [
            part
            for part in path.relative_to(root).parts
            if not part.startswith("popen-gw") and not re.fullmatch(r"test-data\d+", part)
        ]
        files[Path(*parts)] = path
    return files


def _binary_note(relative, base_path, head_path):
    """Return the informational report line for one binary output."""
    if base_path is None or head_path is None:
        return f"- `{relative}`: presence differs (informational)"
    base_size, head_size = base_path.stat().st_size, head_path.stat().st_size
    if base_size != head_size:
        return f"- `{relative}`: size {base_size} != {head_size} bytes (informational)"
    return f"- `{relative}`: present ({base_size} bytes)"


def _file_differences(base_path, head_path):
    """Return the differences of one non-binary output present in base and/or head."""
    if base_path is None or head_path is None:
        return [f"File is missing from {'base' if base_path is None else 'head'}."]
    try:
        return _compare_file(base_path, head_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return [f"comparison failed: {exc}"]


def compare_trees(base_dir, head_dir, report_path):
    """Write a Markdown report and return the compared-output exit status."""
    base_dir, head_dir = Path(base_dir), Path(head_dir)
    base_files, head_files = _relative_files(base_dir), _relative_files(head_dir)
    lines = ["# Differential output report", "", "## Compared outputs", ""]
    failed = not base_dir.is_dir() or not head_dir.is_dir()
    if failed:
        missing_dirs = [str(path) for path in (base_dir, head_dir) if not path.is_dir()]
        lines.append(f"Output directory missing: {', '.join(missing_dirs)}.\n")
    binary_notes = []
    for relative in sorted(base_files.keys() | head_files.keys()):
        base_path, head_path = base_files.get(relative), head_files.get(relative)
        if relative.suffix.lower() in BINARY_SUFFIXES:
            binary_notes.append(_binary_note(relative, base_path, head_path))
            continue
        differences = _file_differences(base_path, head_path)
        if differences:
            failed = True
            lines.extend([f"### `{relative}`", "", *[f"- {item}" for item in differences], ""])
    if not failed:
        lines.append("No differences found.\n")
    lines.extend(["## Binary outputs (informational)", "", *(binary_notes or ["None found."]), ""])
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")
    return 1 if failed else 0


def _pytest_ids(checkout, config_names, model_versions):
    """Resolve listed integration YAML files to pytest parameter IDs."""
    config_dir = checkout / "tests/integration_tests/config"
    ids = []
    for name in config_names:
        path = config_dir / name
        if not path.is_file():
            continue
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        for application in document.get("applications", []):
            configured_version = application.get("configuration", {}).get("model_version")
            if isinstance(configured_version, str):
                configured_versions = {version.strip() for version in configured_version.split(",")}
                if not configured_versions.intersection(model_versions):
                    print(
                        f"Skipping {name} for model version(s) {sorted(model_versions)}: "
                        f"configured for {configured_version}."
                    )
                    continue
            ids.append(
                f"{application.get('application', 'no-app-name')}_"
                f"{application.get('test_name', 'no-test-name')}"
            )
    return ids


def run_integration(
    checkout, basetemp, model_versions, simulation_models_path=None, simtools_tests_path=None
):
    """Run selected application integration cases in a checkout."""
    checkout = Path(checkout).resolve()
    script_dir = Path(__file__).resolve().parent
    list_file = script_dir / "differential_outputs.txt"
    if not list_file.exists():
        list_file = checkout / ".github/scripts/differential_outputs.txt"
    names = [
        line.strip()
        for line in list_file.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    available_names = {
        path.name for path in (checkout / "tests/integration_tests/config").glob("*.yml")
    }
    missing = sorted(set(names) - available_names)
    if missing:
        print(
            f"Listed integration configurations are missing: {', '.join(missing)}", file=sys.stderr
        )
    versions = [version.strip() for version in model_versions.split(",") if version.strip()]
    Path(basetemp).mkdir(parents=True, exist_ok=True)
    return_codes = [
        _run_model_version(
            checkout,
            Path(basetemp) / version,
            version,
            names,
            simulation_models_path,
            simtools_tests_path,
        )
        for version in versions
    ]
    return max(return_codes, default=0)


def _run_model_version(
    checkout, basetemp, model_version, names, simulation_models_path, simtools_tests_path
):
    """Run the listed integration cases for one model version."""
    ids = _pytest_ids(checkout, names, {model_version})
    if not ids:
        print(f"No listed integration configurations match model version {model_version}.")
        return 0
    test_file = "tests/integration_tests/test_applications_from_config.py"
    option_args = ["--model_version", model_version]
    if simulation_models_path:
        option_args.extend(["--simulation_models_path", str(simulation_models_path)])
    if simtools_tests_path:
        option_args.extend(["--simtools_tests_path", str(simtools_tests_path)])
    collect_command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "--no-cov",
        "--collect-only",
        "-qq",
        *option_args,
        test_file,
    ]
    collection = subprocess.run(
        collect_command, cwd=checkout, check=False, capture_output=True, text=True
    )
    if collection.returncode:
        print(collection.stdout, end="", file=sys.stdout)
        print(collection.stderr, end="", file=sys.stderr)
        return collection.returncode
    collected_nodes = [
        line.strip()
        for line in collection.stdout.splitlines()
        if "::test_applications_from_config[" in line
    ]
    expected_ids = set(ids)
    target_nodes = [
        node
        for node in collected_nodes
        if node.rsplit("[", maxsplit=1)[-1].rstrip("]") in expected_ids
    ]
    if not target_nodes:
        print("None of the listed integration configurations are present in this checkout.")
        return 0
    expression = " or ".join(
        "(" + " and ".join(re.findall(r"[A-Za-z0-9_]+", test_id)) + ")" for test_id in ids
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "--no-cov",
        "-n",
        "0",
        "--basetemp",
        str(Path(basetemp).resolve()),
        *option_args,
        test_file,
        "-k",
        expression,
    ]
    target_set = set(target_nodes)
    command.extend(f"--deselect={node}" for node in collected_nodes if node not in target_set)
    return subprocess.run(command, cwd=checkout, check=False).returncode


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="run selected integration tests")
    run_parser.add_argument("--checkout", type=Path, required=True)
    run_parser.add_argument("--basetemp", type=Path, required=True)
    run_parser.add_argument("--model-versions", default="7.0.0")
    run_parser.add_argument("--simulation_models_path", type=Path)
    run_parser.add_argument("--simtools_tests_path", type=Path)
    compare_parser = commands.add_parser("compare", help="compare generated output trees")
    compare_parser.add_argument("--base", type=Path, required=True)
    compare_parser.add_argument("--head", type=Path, required=True)
    compare_parser.add_argument("--report", type=Path, default=Path("diff_report.md"))
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_integration(
            args.checkout,
            args.basetemp,
            args.model_versions,
            args.simulation_models_path,
            args.simtools_tests_path,
        )
    return compare_trees(args.base, args.head, args.report)


if __name__ == "__main__":
    raise SystemExit(main())

"""Discover, filter, group, and validate production metadata manifests."""

import hashlib
import json
import re
from dataclasses import dataclass, field
from math import isclose
from pathlib import Path

import astropy.units as u

from simtools.constants import SCHEMA_PATH
from simtools.data_model import schema
from simtools.io import ascii_handler
from simtools.simtel.simtel_io_metadata import read_sim_telarray_metadata

SIMULATE_PROD_JOB_METADATA = "simulate_prod_job_metadata.yml"
SUPPORTED_SCHEMA_VERSIONS = {"1.0.0"}

_RUN_NUMBER_PATTERN = re.compile(r"(?:^|_)run0*([0-9]+)(?:_|\.|$)")
_FILE_TYPE_SUFFIXES = {
    "reduced_event_data": (".reduced_event_data.hdf5",),
    "sim_telarray": (".simtel.zst", ".simtel.gz", ".simtel"),
    "sim_telarray_log": (".simtel.log.gz", ".simtel.log"),
    "sim_telarray_histogram": (".histogram.hdf5", ".histogram.gz", ".histogram"),
    "corsika": (".corsika.zst", ".corsika.gz", ".corsika"),
    "corsika_log": (".corsika.log.gz", ".corsika.log"),
    "trigger_histograms": (".trigger_histograms.hdf5",),
}
_MANIFEST_SCHEMAS = {
    "simulate_prod_job": "simulate_prod_job_metadata.schema.yml",
    "trigger_histograms": "trigger_histograms_metadata.schema.yml",
}
_GROUPING_EXCLUDE_KEYS = {
    "run_number",
    "random_seed",
    "corsika_seeds",
    "sim_telarray_seed",
    "sim_telarray_seed_file",
    "sim_telarray_instrument_seed",
    "sim_telarray_random_instrument_instances",
    "output_path",
    "grid_output_path",
}
_SIMTEL_METADATA_MANIFEST_KEYS = {
    "primary": ("primary", "primary_particle", "particle"),
    "azimuth_angle": ("azimuth", "azimuth_angle"),
    "zenith_angle": ("zenith", "zenith_angle"),
    "energy_min": ("energy_min",),
    "energy_max": ("energy_max",),
    "view_cone_min": ("viewcone_min", "view_cone_min"),
    "view_cone_max": ("viewcone_max", "view_cone_max"),
    "core_scatter_max": ("core_scatter_max",),
}


@dataclass(frozen=True)
class ProductionManifest:
    """A loaded production metadata manifest and its file path."""

    path: Path
    data: dict

    @property
    def directory(self):
        """Return the manifest parent directory."""
        return self.path.parent

    @property
    def run_number(self):
        """Return the run number recorded in this manifest."""
        return int(self.data.get("configuration", {})["run_number"])


@dataclass
class ProductionFileGroup:
    """Selected production files sharing one simulation configuration."""

    configuration: dict
    run_numbers: list[int] = field(default_factory=list)
    file_paths: list[Path] = field(default_factory=list)
    missing_run_numbers: list[int] = field(default_factory=list)
    duplicate_run_numbers: list[int] = field(default_factory=list)


def find_manifests(production_path, manifest_name=SIMULATE_PROD_JOB_METADATA):
    """Return sorted production metadata manifests below a directory."""
    production_path = Path(production_path)
    if not production_path.is_dir():
        raise FileNotFoundError(f"Production path not found: {production_path}")
    return sorted(path for path in production_path.rglob(manifest_name) if path.is_file())


def load_manifest(manifest_path):
    """Load and validate one production manifest YAML file."""
    manifest_path = Path(manifest_path)
    data = ascii_handler.collect_data_from_file(manifest_path)
    _validate_manifest_structure(data, manifest_path)
    return ProductionManifest(path=manifest_path, data=data)


def discover_manifests(production_path, manifest_name=SIMULATE_PROD_JOB_METADATA):
    """Load all production metadata manifests below a directory."""
    return [load_manifest(path) for path in find_manifests(production_path, manifest_name)]


def discover_product_manifests(production_path, product_type, manifest_pattern="*.yml"):
    """Load all manifests of one product type below a directory."""
    production_path = Path(production_path)
    if not production_path.is_dir():
        raise FileNotFoundError(f"Production path not found: {production_path}")
    manifests = []
    for path in sorted(production_path.rglob(manifest_pattern)):
        data = ascii_handler.collect_data_from_file(path)
        if not isinstance(data, dict) or data.get("product_type") != product_type:
            continue
        _validate_manifest_structure(data, path)
        manifests.append(ProductionManifest(path=path, data=data))
    return manifests


def select_file_groups(
    production_path,
    selections=None,
    file_type="reduced_event_data",
    require_complete_runs=False,
    manifest_name=SIMULATE_PROD_JOB_METADATA,
):
    """Discover manifests, filter jobs, and group selected files by configuration."""
    manifests = discover_manifests(production_path, manifest_name=manifest_name)
    selected = filter_manifests(manifests, selections or [])
    groups = group_selected_files(selected, file_type=file_type)
    if require_complete_runs:
        incomplete = [group for group in groups if group.missing_run_numbers]
        if incomplete:
            raise ValueError(
                "Missing run numbers in selected production groups: "
                + "; ".join(",".join(map(str, group.missing_run_numbers)) for group in incomplete)
            )
    return {
        "metadata_files_read": len(manifests),
        "matching_jobs": len(selected),
        "configuration_groups": len(groups),
        "groups": groups,
    }


def filter_manifests(manifests, selections):
    """Return manifests matching all selection expressions."""
    parsed = [_parse_selection(selection) for selection in selections]
    return [
        manifest
        for manifest in manifests
        if all(_selection_matches(manifest.data, key, expected) for key, expected in parsed)
    ]


def group_selected_files(manifests, file_type="reduced_event_data"):
    """Group selected files by all configuration fields except the run number."""
    groups = {}
    for manifest in manifests:
        check_manifest(manifest)
        files = _manifest_files(manifest, file_type)
        if not files:
            raise ValueError(f"Manifest {manifest.path} lists no files of type '{file_type}'.")
        key = _normalized_group_key(manifest.data.get("configuration", {}))
        group = groups.setdefault(
            key,
            ProductionFileGroup(
                configuration=_configuration_without_excluded_keys(
                    manifest.data.get("configuration", {})
                )
            ),
        )
        for file_path in files:
            group.run_numbers.append(manifest.run_number)
            group.file_paths.append(file_path)

    grouped = list(groups.values())
    for group in grouped:
        _sort_group_by_run_number(group)
        group.duplicate_run_numbers = _duplicate_run_numbers(group.run_numbers)
        if group.duplicate_run_numbers:
            raise ValueError(
                "Duplicate run numbers in selected production group: "
                + ", ".join(map(str, group.duplicate_run_numbers))
            )
        group.missing_run_numbers = _missing_run_numbers(group.run_numbers)
    return grouped


def check_manifest(manifest):
    """Validate a manifest against its output directory and listed files."""
    if not isinstance(manifest, ProductionManifest):
        manifest = load_manifest(manifest)

    _validate_manifest_structure(manifest.data, manifest.path)
    seen_paths = set()
    unverifiable_fields = sorted(manifest.data.get("configuration", {}))
    for file_type, relative_paths in manifest.data.get("files", {}).items():
        for relative_path in relative_paths:
            file_path = _resolve_relative_manifest_path(manifest.directory, relative_path)
            if file_path in seen_paths:
                raise ValueError(
                    f"Duplicate output file listed in {manifest.path}: {relative_path}"
                )
            seen_paths.add(file_path)
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Manifest {manifest.path} references missing file: {relative_path}"
                )
            _validate_file_type(file_path, file_type, manifest.path)
            _validate_filename_run_number(file_path, manifest)
            if file_type == "sim_telarray":
                unverifiable_fields = _compare_simtel_metadata(
                    file_path,
                    manifest,
                    unverifiable_fields,
                )
    if manifest.data["product_type"] == "simulate_prod_job":
        _check_declared_production_inventory(manifest, seen_paths)

    return {"valid": True, "unverifiable_fields": unverifiable_fields}


def inventory_production_files(job_directory):
    """Return nested production outputs grouped by manifest file type.

    Paths in the returned inventory are relative to ``job_directory`` so that
    manifests can describe the standard ``sim_telarray/runNNNNNN`` and
    ``corsika/runNNNNNN`` output layout.
    """
    job_directory = Path(job_directory)
    inventory = {}
    for file_path in sorted(path for path in job_directory.rglob("*") if path.is_file()):
        file_type = _production_file_type(file_path)
        if file_type is not None:
            relative_path = file_path.relative_to(job_directory).as_posix()
            inventory.setdefault(file_type, []).append(relative_path)
    return inventory


def validate_required_production_outputs(file_inventory, simulation_software, job_directory):
    """Require the principal simulation output before marking a job complete."""
    required_type = "corsika" if simulation_software == "corsika" else "sim_telarray"
    if not file_inventory.get(required_type):
        raise ValueError(
            f"Incomplete production job {job_directory}: no '{required_type}' output file found."
        )


def stable_configuration_hash(value, length=8):
    """Return a short stable hash for normalized configuration data."""
    normalized = normalize_for_comparison(value)
    payload = json.dumps(normalized, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:length]


def write_selection_file(selection_result, output_file):
    """Write selected file groups to a YAML file."""
    output = {
        "schema_version": "1.0.0",
        "product_type": "production_file_selection",
        "metadata_files_read": selection_result["metadata_files_read"],
        "matching_jobs": selection_result["matching_jobs"],
        "configuration_groups": selection_result["configuration_groups"],
        "groups": [
            {
                "configuration": group.configuration,
                "run_numbers": group.run_numbers,
                "missing_run_numbers": group.missing_run_numbers,
                "files": [str(path) for path in group.file_paths],
            }
            for group in selection_result["groups"]
        ],
    }
    ascii_handler.write_data_to_file(output, output_file)


def selection_summary(selection_result):
    """Return a compact summary string for logging or console output."""
    missing = sorted(
        {
            run_number
            for group in selection_result["groups"]
            for run_number in group.missing_run_numbers
        }
    )
    return (
        f"Metadata files read: {selection_result['metadata_files_read']}\n"
        f"Matching jobs: {selection_result['matching_jobs']}\n"
        f"Configuration groups: {selection_result['configuration_groups']}\n"
        f"Missing runs: {', '.join(map(str, missing)) if missing else 'none'}"
    )


def normalize_for_comparison(value):
    """Normalize quantities and containers for exact matching and grouping."""
    if isinstance(value, u.Quantity):
        return _quantity_comparison_value(value.value, value.unit)
    if isinstance(value, dict) and set(value) == {"value", "unit"}:
        return _quantity_comparison_value(value["value"], value["unit"])
    if isinstance(value, dict):
        return tuple(
            (key, normalize_for_comparison(value[key]))
            for key in sorted(value)
            if key not in _GROUPING_EXCLUDE_KEYS
        )
    if isinstance(value, list | tuple):
        return tuple(normalize_for_comparison(item) for item in value)
    return value


def _validate_manifest_structure(data, manifest_path):
    """Validate required manifest fields and supported version."""
    if not isinstance(data, dict):
        raise ValueError(f"Malformed metadata in {manifest_path}: expected a mapping.")
    for key in ("schema_version", "product_type", "status", "configuration", "files"):
        if key not in data:
            raise ValueError(f"Malformed metadata in {manifest_path}: missing '{key}'.")
    if str(data["schema_version"]) not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported production metadata schema version in {manifest_path}: "
            f"{data['schema_version']}"
        )
    if data["status"] != "complete":
        raise ValueError(f"Production job is not complete in {manifest_path}.")
    if not isinstance(data["configuration"], dict):
        raise ValueError(f"Malformed metadata in {manifest_path}: configuration is not a mapping.")
    if data["product_type"] == "simulate_prod_job" and "run_number" not in data["configuration"]:
        raise ValueError(
            f"Malformed metadata in {manifest_path}: missing configuration.run_number."
        )
    if not isinstance(data["files"], dict):
        raise ValueError(f"Malformed metadata in {manifest_path}: files is not a mapping.")
    schema_file = _MANIFEST_SCHEMAS.get(data["product_type"])
    if schema_file is not None:
        schema.validate_dict_using_schema(
            ascii_handler.to_builtin(data),
            schema_file=SCHEMA_PATH / schema_file,
            offline=True,
        )


def _parse_selection(selection):
    """Parse a KEY=VALUE selection expression."""
    if "=" not in selection:
        raise ValueError(f"Selection must use KEY=VALUE syntax: {selection}")
    key, expected = selection.split("=", maxsplit=1)
    if not key:
        raise ValueError(f"Selection key is empty: {selection}")
    return key, expected.strip().strip("\"'")


def _selection_matches(data, key, expected):
    """Return whether one dotted-path selection matches."""
    value = _get_dotted_value(data, key)
    if value is None and "." not in key:
        value = _get_dotted_value(data, f"configuration.{key}")
    return _values_match(value, expected)


def _get_dotted_value(data, key):
    """Return a nested value addressed by a dotted key path."""
    value = data
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _values_match(value, expected):
    """Return whether a manifest value matches a command-line selection value."""
    if value is None:
        return False
    if isinstance(value, dict) and set(value) == {"value", "unit"}:
        return _quantity_matches(value, expected)
    return str(value) == expected


def _quantity_matches(value, expected):
    """Return whether a stored quantity matches a textual quantity exactly after conversion."""
    stored = float(value["value"]) * u.Unit(value["unit"])
    try:
        expected_quantity = u.Quantity(expected)
    except TypeError, ValueError:
        return str(value) == expected
    if expected_quantity.unit == u.dimensionless_unscaled:
        expected_quantity = expected_quantity.value * stored.unit
    return stored.unit.is_equivalent(expected_quantity.unit) and isclose(
        stored.to_value(expected_quantity.unit), expected_quantity.value
    )


def _quantity_comparison_value(value, unit):
    """Return a normalized comparable representation of a quantity."""
    quantity = float(value) * u.Unit(unit)
    return (round(float(quantity.si.value), 12), str(quantity.si.unit))


def _configuration_without_excluded_keys(configuration):
    """Return configuration fields used for grouping."""
    return {key: value for key, value in configuration.items() if key not in _GROUPING_EXCLUDE_KEYS}


def _normalized_group_key(configuration):
    """Return a stable normalized grouping key for a configuration."""
    return normalize_for_comparison(_configuration_without_excluded_keys(configuration))


def _manifest_files(manifest, file_type):
    """Return absolute paths for files of the requested manifest file type."""
    return [
        _resolve_relative_manifest_path(manifest.directory, relative_path)
        for relative_path in manifest.data.get("files", {}).get(file_type, [])
    ]


def _resolve_relative_manifest_path(directory, relative_path):
    """Resolve one manifest-relative path and reject paths escaping the job directory."""
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"Manifest file path must be relative: {relative_path}")
    resolved = (directory / path).resolve()
    try:
        resolved.relative_to(directory.resolve())
    except ValueError as exc:
        raise ValueError(f"Manifest file path escapes job directory: {relative_path}") from exc
    return resolved


def _validate_file_type(file_path, file_type, manifest_path):
    """Validate a file suffix for known manifest file types."""
    suffixes = _FILE_TYPE_SUFFIXES.get(file_type)
    if suffixes and not any(file_path.name.endswith(suffix) for suffix in suffixes):
        raise ValueError(
            f"Manifest {manifest_path} lists {file_path.name} as '{file_type}', "
            "but its suffix is inconsistent."
        )


def _production_file_type(file_path):
    """Return the manifest type for a recognized production output file."""
    for file_type, suffixes in _FILE_TYPE_SUFFIXES.items():
        if file_type == "trigger_histograms":
            continue
        if any(file_path.name.endswith(suffix) for suffix in suffixes):
            return file_type
    return None


def _check_declared_production_inventory(manifest, declared_paths):
    """Require the manifest to list every recognized production output."""
    actual_paths = {
        (manifest.directory / relative_path).resolve()
        for paths in inventory_production_files(manifest.directory).values()
        for relative_path in paths
    }
    unexpected = sorted(path.name for path in actual_paths - declared_paths)
    if unexpected:
        raise ValueError(
            f"Unexpected production files not listed in {manifest.path}: " + ", ".join(unexpected)
        )


def _validate_filename_run_number(file_path, manifest):
    """Validate encoded run numbers when the filename contains one."""
    if "run_number" not in manifest.data.get("configuration", {}):
        return
    match = _RUN_NUMBER_PATTERN.search(file_path.name)
    if match and int(match.group(1)) != manifest.run_number:
        raise ValueError(
            f"Run number mismatch for {file_path.name}: filename encodes "
            f"{int(match.group(1))}, manifest records {manifest.run_number}."
        )


def _compare_simtel_metadata(file_path, manifest, unverifiable_fields):
    """Compare manifest configuration with matching embedded sim_telarray metadata."""
    try:
        metadata, _ = read_sim_telarray_metadata(file_path)
    except OSError, ValueError, AttributeError:
        return unverifiable_fields

    remaining = set(unverifiable_fields)
    configuration = manifest.data.get("configuration", {})
    for manifest_key, metadata_keys in _SIMTEL_METADATA_MANIFEST_KEYS.items():
        if manifest_key not in configuration:
            continue
        metadata_key = next((key for key in metadata_keys if key in metadata), None)
        if metadata_key is None:
            continue
        if not _embedded_metadata_matches(configuration[manifest_key], metadata[metadata_key]):
            raise ValueError(
                f"Manifest {manifest.path} field configuration.{manifest_key} does not match "
                f"embedded sim_telarray metadata in {file_path.name}."
            )
        remaining.discard(manifest_key)
    return sorted(remaining)


def _embedded_metadata_matches(manifest_value, metadata_value):
    """Return whether one embedded metadata value matches a manifest value."""
    if isinstance(manifest_value, u.Quantity):
        manifest_value = {"value": manifest_value.value, "unit": manifest_value.unit}
    if isinstance(manifest_value, dict) and set(manifest_value) == {"value", "unit"}:
        return _quantity_matches(manifest_value, str(metadata_value))
    return str(manifest_value).lower() == str(metadata_value).lower()


def _sort_group_by_run_number(group):
    """Sort grouped run numbers and files by run number."""
    ordered = sorted(zip(group.run_numbers, group.file_paths), key=lambda item: item[0])
    group.run_numbers = [run_number for run_number, _ in ordered]
    group.file_paths = [file_path for _, file_path in ordered]


def _duplicate_run_numbers(run_numbers):
    """Return sorted duplicate run numbers."""
    seen = set()
    duplicates = set()
    for run_number in run_numbers:
        if run_number in seen:
            duplicates.add(run_number)
        seen.add(run_number)
    return sorted(duplicates)


def _missing_run_numbers(run_numbers):
    """Return missing run numbers between the minimum and maximum selected run."""
    if not run_numbers:
        return []
    present = set(run_numbers)
    return [
        run_number
        for run_number in range(min(present), max(present) + 1)
        if run_number not in present
    ]

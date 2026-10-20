"""Read and compare camera-efficiency result artifacts."""

import hashlib
import json
import math
from pathlib import Path

import astropy.units as u
import numpy as np
import yaml

from simtools.version import __version__

_TELESCOPE_KEY = "telescope"
_ZENITH_KEY = "zenith_angle"
_AZIMUTH_KEY = "azimuth_angle"
_NSB_SPECTRUM_KEY = "nsb_spectrum"
_SKIP_NSB_CORRECTION_KEY = "skip_correction_to_nsb_spectrum"
_MODEL_VERSION_KEY = "model_version"
_METRICS_KEY = "metrics"
_DIAGNOSTICS_KEY = "diagnostics"
_SCHEMA_VERSION_KEY = "schema_version"
_STATUS_KEY = "status"
_INPUTS_KEY = "inputs"
_ARTIFACTS_KEY = "artifacts"
_SUMMARY_KEY = "summary"
_BASELINE_KEY = "baseline"
_CANDIDATE_KEY = "candidate"
_ROOT_KEY = "_root"
_VALUE_KEY = "value"
_COMPLETE_STATUS = "complete"
_INCOMPLETE_STATUS = "incomplete"
_ENCODING = "utf-8"
_SPECTRUM_KEY = "spectrum"
_SHA256_KEY = "sha256"
_SHOWER_CHANNEL = "shower"

_METRIC_UNITS = {
    "reflectivity": "",
    "cam_eff": "",
    "tel_eff": "",
    "tot_sens": "",
    "nsb_rate": "GHz",
    "nsb_ref": "GHz",
    "muon_frac": "",
}
_CASE_FIELDS = (
    "site",
    _TELESCOPE_KEY,
    _ZENITH_KEY,
    _AZIMUTH_KEY,
    _NSB_SPECTRUM_KEY,
    _SKIP_NSB_CORRECTION_KEY,
    _MODEL_VERSION_KEY,
)
_CASE_KEY = "case"
_SOFTWARE_KEY = "software"
_IMAGE_KEY = "image"
_SIMTOOLS_VERSION_KEY = "simtools_version"
_DEFAULT_NSB_SPECTRUM = "default sim_telarray spectrum"
_WAVELENGTH_CHANNELS = (_SHOWER_CHANNEL, "muon", "nsb")


def read_summary(path):
    """Normalize a legacy camera-efficiency YAML summary."""
    summary = yaml.safe_load(Path(path).read_text(encoding=_ENCODING))
    if not isinstance(summary, dict):
        raise ValueError(f"Camera-efficiency summary must be a mapping: {path}")
    metadata = summary.get("meta", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Camera-efficiency metadata must be a mapping: {path}")
    case = {
        _TELESCOPE_KEY: metadata.get("tel"),
        _MODEL_VERSION_KEY: metadata.get(_MODEL_VERSION_KEY),
        _ZENITH_KEY: _angle_value(metadata.get("zen")),
        _AZIMUTH_KEY: _angle_value(metadata.get("az")),
        _NSB_SPECTRUM_KEY: metadata.get("nsb"),
    }
    return {_CASE_KEY: case, **normalize_summary(summary)}


def normalize_summary(summary):
    """Normalize a parsed summary mapping to numeric metric values and units."""
    metrics = {}
    diagnostics = []
    for name, unit in _METRIC_UNITS.items():
        if name not in summary:
            metrics[name] = None
            diagnostics.append(f"Missing metric {name}.")
            continue
        try:
            metric = summary[name]
            value = _normalize_metric(metric, unit, name)
        except (KeyError, TypeError, ValueError) as exc:
            metrics[name] = None
            diagnostics.append(str(exc))
            continue
        metrics[name] = {_VALUE_KEY: value, "unit": unit}
    return {_METRICS_KEY: metrics, _DIAGNOSTICS_KEY: diagnostics}


def create_manifest(
    result,
    config,
    output_directory,
    artifacts=None,
    image_identity=None,
    exit_status=0,
    simtools_version=None,
):
    """Create a compact versioned manifest around calculated summary results."""
    output_root = Path(output_directory).resolve()
    case = {field: _config_value(config, field) for field in _CASE_FIELDS}
    inputs = {}
    diagnostics = list(result.get(_DIAGNOSTICS_KEY, []))
    spectrum = config.get(_NSB_SPECTRUM_KEY)
    if spectrum and spectrum != _DEFAULT_NSB_SPECTRUM:
        spectrum_path = Path(spectrum).resolve()
        if spectrum_path.is_file():
            inputs[_SPECTRUM_KEY] = {
                "path": spectrum_path.name,
                _SHA256_KEY: sha256_file(spectrum_path),
            }
        else:
            diagnostics.append(f"NSB spectrum input was unavailable for hashing: {spectrum_path}")
    artifact_paths = {
        name: _relative_artifact_path(value, output_root)
        for name, value in (artifacts or {}).items()
    }
    if not image_identity:
        diagnostics.append("Container image identity was not provided.")
    if not simtools_version:
        diagnostics.append("Actual simtools version was not available from the run.")
    return {
        _SCHEMA_VERSION_KEY: 1,
        _STATUS_KEY: (
            _COMPLETE_STATUS if exit_status == 0 and not diagnostics else _INCOMPLETE_STATUS
        ),
        _CASE_KEY: case,
        _SOFTWARE_KEY: {
            _SIMTOOLS_VERSION_KEY: simtools_version or __version__,
            _MODEL_VERSION_KEY: case[_MODEL_VERSION_KEY],
        },
        _IMAGE_KEY: image_identity,
        _INPUTS_KEY: inputs,
        _METRICS_KEY: result.get(_METRICS_KEY, {}),
        _ARTIFACTS_KEY: artifact_paths,
        "exit_status": exit_status,
        _DIAGNOSTICS_KEY: diagnostics,
    }


def collect_legacy_results(output_directory, config, image_identity, simtools_version):
    """Create a manifest from outputs produced by a legacy validator command."""
    output = Path(output_directory).resolve()
    summary_path = _summary_path(output)
    result = read_summary(summary_path)
    effective_config = dict(config)
    for key in (_TELESCOPE_KEY, _MODEL_VERSION_KEY):
        if result[_CASE_KEY].get(key):
            effective_config[key] = result[_CASE_KEY][key]
    for key in (_ZENITH_KEY, _AZIMUTH_KEY):
        if effective_config.get(key) is None:
            effective_config[key] = result[_CASE_KEY].get(key)
    if not effective_config.get(_NSB_SPECTRUM_KEY):
        effective_config[_NSB_SPECTRUM_KEY] = result[_CASE_KEY].get(_NSB_SPECTRUM_KEY)
    effective_config.setdefault(_SKIP_NSB_CORRECTION_KEY, False)
    tables = sorted(output.glob("*.ecsv"))
    plots = sorted(
        path for path in output.iterdir() if path.suffix.lower() in {".png", ".pdf", ".svg"}
    )
    artifacts = {_SUMMARY_KEY: summary_path, "wavelength_tables": tables, "native_plots": plots}
    for channel in _WAVELENGTH_CHANNELS:
        table = next((path for path in tables if channel in path.name.lower()), None)
        if table:
            artifacts[f"{channel}_wavelength_table"] = table
            if channel == _SHOWER_CHANNEL:
                artifacts["wavelength_table"] = table
    return create_manifest(
        result,
        effective_config,
        output,
        artifacts=artifacts,
        image_identity=image_identity,
        simtools_version=simtools_version,
    )


def sha256_file(path):
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_manifests(baseline, candidate, compare_by=None, tolerances=None):
    """Compare result manifests without running a calculation."""
    compare_by = set(compare_by or [])
    tolerances = tolerances or {}
    diagnostics = _compatibility_diagnostics(baseline, candidate, compare_by)
    diagnostics.extend(_provenance_diagnostics(baseline, _BASELINE_KEY))
    diagnostics.extend(_provenance_diagnostics(candidate, _CANDIDATE_KEY))
    changes, metric_diagnostics = _compare_metrics(baseline, candidate, tolerances)
    diagnostics.extend(metric_diagnostics)
    diagnostics.extend(baseline.get(_DIAGNOSTICS_KEY, []))
    diagnostics.extend(candidate.get(_DIAGNOSTICS_KEY, []))
    if (
        baseline.get(_STATUS_KEY) != _COMPLETE_STATUS
        or candidate.get(_STATUS_KEY) != _COMPLETE_STATUS
    ):
        diagnostics.append("One or both input manifests are incomplete.")
    return {
        _SCHEMA_VERSION_KEY: 1,
        _STATUS_KEY: _INCOMPLETE_STATUS if diagnostics else _COMPLETE_STATUS,
        _BASELINE_KEY: _comparison_identity(baseline),
        _CANDIDATE_KEY: _comparison_identity(candidate),
        "changes": changes,
        _DIAGNOSTICS_KEY: list(dict.fromkeys(diagnostics)),
        "plots": {},
    }


def _comparison_identity(manifest):
    """Retain case and runtime identity in the comparison artifact."""
    case = manifest.get(_CASE_KEY)
    identity = case if isinstance(case, dict) else {}
    return identity | {
        _SOFTWARE_KEY: manifest.get(_SOFTWARE_KEY, {}),
        _IMAGE_KEY: manifest.get(_IMAGE_KEY),
    }


def _provenance_diagnostics(manifest, label):
    """Reject complete-marked manifests that omit required case or runtime evidence."""
    diagnostics = []
    case = manifest.get(_CASE_KEY)
    case = case if isinstance(case, dict) else {}
    prefix = f"Missing {label}"
    for field in _CASE_FIELDS:
        if case.get(field) is None:
            diagnostics.append(f"{prefix} case provenance: {field}.")
    software = manifest.get(_SOFTWARE_KEY)
    if not isinstance(software, dict) or not software.get(_SIMTOOLS_VERSION_KEY):
        diagnostics.append(f"{prefix} simtools version.")
    if not manifest.get(_IMAGE_KEY):
        diagnostics.append(f"{prefix} image identity.")
    spectrum = case.get(_NSB_SPECTRUM_KEY)
    if spectrum != _DEFAULT_NSB_SPECTRUM:
        inputs = manifest.get(_INPUTS_KEY)
        spectrum_input = inputs.get(_SPECTRUM_KEY, {}) if isinstance(inputs, dict) else {}
        if not isinstance(spectrum_input, dict) or not spectrum_input.get(_SHA256_KEY):
            diagnostics.append(f"{prefix} spectrum input hash.")
    return diagnostics


def _compatibility_diagnostics(baseline, candidate, compare_by):
    """Find undeclared case-setting and input-hash differences."""
    diagnostics = []
    case_paths = {field: f"case.{field}" for field in _CASE_FIELDS}
    input_names = set(baseline.get(_INPUTS_KEY, {})) | set(candidate.get(_INPUTS_KEY, {}))
    input_paths = {name: f"inputs.{name}.sha256" for name in input_names}
    known_paths = set(case_paths.values()) | set(input_paths.values())
    for path in compare_by - known_paths:
        diagnostics.append(f"Unsupported declared comparison field {path}.")
    for field, path in case_paths.items():
        if path in compare_by:
            continue
        if baseline.get(_CASE_KEY, {}).get(field) != candidate.get(_CASE_KEY, {}).get(field):
            diagnostics.append(f"Incompatible case setting {path}.")
    for name, path in input_paths.items():
        if path not in compare_by:
            left = baseline.get(_INPUTS_KEY, {}).get(name, {}).get(_SHA256_KEY)
            right = candidate.get(_INPUTS_KEY, {}).get(name, {}).get(_SHA256_KEY)
            if left != right:
                diagnostics.append(f"Incompatible input hash {path}.")
    return diagnostics


def _compare_metrics(baseline, candidate, tolerances):
    """Compare each declared metric and gather per-metric diagnostics."""
    changes = {}
    diagnostics = []
    baseline_metrics = baseline.get(_METRICS_KEY, {})
    candidate_metrics = candidate.get(_METRICS_KEY, {})
    for name in _METRIC_UNITS:
        change, diagnostic = _metric_change(
            name,
            baseline_metrics.get(name),
            candidate_metrics.get(name),
            tolerances.get(name),
        )
        if diagnostic:
            diagnostics.append(diagnostic)
        if change is not None:
            changes[name] = change
    return changes, diagnostics


def _metric_change(name, left, right, tolerance):
    """Return one normalized metric delta or its diagnostic."""
    if left is None or right is None:
        return None, f"Missing or invalid metric {name}."
    try:
        before = _finite_number(left[_VALUE_KEY], name)
        after = _finite_number(right[_VALUE_KEY], name)
        source_unit = u.Unit(right.get("unit", ""))
        target_unit = u.Unit(left.get("unit", ""))
        after = _finite_number((after * source_unit).to_value(target_unit), name)
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"Incompatible metric {name}: {exc}"
    delta = after - before
    relative = None if before == 0 else delta / before
    if not math.isfinite(delta) or (relative is not None and not math.isfinite(relative)):
        return None, f"Non-finite change for metric {name}."
    return {
        _BASELINE_KEY: {_VALUE_KEY: before, "unit": left.get("unit", "")},
        _CANDIDATE_KEY: {_VALUE_KEY: after, "unit": left.get("unit", "")},
        "absolute": delta,
        "relative": relative,
        "relative_reason": "baseline is zero" if before == 0 else None,
        "tolerance": tolerance,
        "advisory": relative is not None and tolerance is not None and abs(relative) > tolerance,
    }, None


def write_json(path, value):
    """Write JSON with numeric values from NumPy and Astropy objects normalized."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, default=_json_value) + "\n", encoding=_ENCODING)


def load_manifest(path):
    """Load a manifest or adapt a legacy YAML summary with unknown provenance."""
    result_path = Path(path)
    if result_path.is_dir():
        manifest_path = result_path / "camera_efficiency_manifest.json"
        if manifest_path.is_file():
            result_path = manifest_path
        else:
            result_path = _summary_path(result_path)
    if result_path.suffix.lower() in {".yml", ".yaml"}:
        summary = read_summary(result_path)
        summary.update(
            {
                _SCHEMA_VERSION_KEY: 1,
                _STATUS_KEY: _INCOMPLETE_STATUS,
                _SOFTWARE_KEY: {},
                _IMAGE_KEY: None,
                _INPUTS_KEY: {},
                _ARTIFACTS_KEY: {_SUMMARY_KEY: result_path.name},
                _DIAGNOSTICS_KEY: summary[_DIAGNOSTICS_KEY]
                + ["Legacy summary does not contain complete case provenance."],
            }
        )
        summary[_ROOT_KEY] = result_path.parent
        return summary
    manifest = json.loads(result_path.read_text(encoding=_ENCODING))
    if not isinstance(manifest, dict) or manifest.get(_SCHEMA_VERSION_KEY) != 1:
        raise ValueError(f"Unsupported camera-efficiency manifest: {result_path}")
    _validate_manifest_mappings(manifest, result_path)
    manifest[_ROOT_KEY] = result_path.parent
    return manifest


def _summary_path(output):
    """Require one combined legacy summary within a telescope's output directory."""
    summaries = sorted(output.glob("*camera_efficiency_summary*.yml"))
    if len(summaries) != 1:
        raise ValueError(f"Expected one camera-efficiency summary in {output}.")
    return summaries[0]


def _validate_manifest_mappings(manifest, path):
    """Reject malformed JSON containers before using them as result records."""
    for field in (_CASE_KEY, _SOFTWARE_KEY, _INPUTS_KEY, _METRICS_KEY, _ARTIFACTS_KEY):
        if not isinstance(manifest.get(field, {}), dict):
            raise ValueError(f"Manifest field {field} must be a mapping: {path}")
    if any(not isinstance(value, dict) for value in manifest.get(_INPUTS_KEY, {}).values()):
        raise ValueError(f"Manifest input records must be mappings: {path}")
    for artifact in manifest.get(_ARTIFACTS_KEY, {}).values():
        paths = artifact if isinstance(artifact, list) else [artifact]
        if any(not isinstance(item, str) for item in paths):
            raise ValueError(f"Manifest artifacts must contain paths or lists of paths: {path}")
    diagnostics = manifest.get(_DIAGNOSTICS_KEY, [])
    if not isinstance(diagnostics, list) or any(not isinstance(item, str) for item in diagnostics):
        raise ValueError(f"Manifest diagnostics must be a list of strings: {path}")


def _config_value(config, name):
    value = config.get(name)
    if name == _SKIP_NSB_CORRECTION_KEY and value is None:
        return False
    if isinstance(value, u.Quantity):
        value = value.to_value("deg")
    if name == _NSB_SPECTRUM_KEY and value:
        value = Path(value).name
    elif name == _NSB_SPECTRUM_KEY and value is None:
        value = _DEFAULT_NSB_SPECTRUM
    return _plain_value(value)


def _relative_artifact_path(value, root):
    if isinstance(value, list):
        return [_relative_artifact_path(item, root) for item in value]
    return Path(value).resolve().relative_to(root).as_posix()


def _plain_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, _VALUE_KEY):
        return _plain_value(value.value)
    return value


def _angle_value(value):
    if isinstance(value, dict) and {_VALUE_KEY, "unit"} <= value.keys():
        return (value[_VALUE_KEY] * u.Unit(value["unit"])).to_value(u.deg)
    if isinstance(value, u.Quantity):
        return value.to_value(u.deg)
    return _plain_value(value)


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        raise ValueError(f"Invalid numeric metric {name}: {value!r}.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Non-finite metric {name}: {numeric}.")
    return numeric


def _metric_value(value, unit, name):
    if isinstance(value, u.Quantity):
        value = value.to_value(unit)
    return _finite_number(value, name)


def _normalize_metric(metric, expected_unit, name):
    value = metric[_VALUE_KEY]
    if isinstance(value, dict):
        return _normalize_metric(value, expected_unit, name)
    if isinstance(value, u.Quantity):
        return _metric_value(value, expected_unit, name)
    number = _finite_number(value, name)
    unit = u.Unit(metric.get("unit", expected_unit))
    return (number * unit).to_value(expected_unit)


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, u.Quantity):
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

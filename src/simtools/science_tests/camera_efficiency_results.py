"""Read and compare camera-efficiency result artifacts."""

import hashlib
import json
import logging
import math
from pathlib import Path

import astropy.units as u
import numpy as np
import yaml

from simtools.version import __version__

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
    "telescope",
    "zenith_angle",
    "azimuth_angle",
    "nsb_spectrum",
    "skip_correction_to_nsb_spectrum",
    "model_version",
)
logger = logging.getLogger(__name__)


def read_summary(path):
    """Normalize a legacy camera-efficiency YAML summary."""
    summary = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"Camera-efficiency summary must be a mapping: {path}")
    metadata = summary.get("meta", {})
    case = {
        "telescope": metadata.get("tel"),
        "model_version": metadata.get("model_version"),
        "zenith_angle": _angle_value(metadata.get("zen")),
        "azimuth_angle": _angle_value(metadata.get("az")),
        "nsb_spectrum": metadata.get("nsb"),
    }
    metrics = {}
    diagnostics = []
    for name, unit in _METRIC_UNITS.items():
        if name not in summary:
            diagnostics.append(f"Missing metric {name}.")
            metrics[name] = None
            continue
        try:
            metric = summary[name]
            value = _normalize_metric(metric, unit, name)
        except (KeyError, TypeError, ValueError) as exc:
            diagnostics.append(str(exc))
            metrics[name] = None
            continue
        metrics[name] = {"value": value, "unit": unit}
    return {"case": case, "metrics": metrics, "diagnostics": diagnostics}


def normalize_summary(summary):
    """Normalize a parsed summary mapping to numeric metric values and units."""
    metrics = {}
    diagnostics = []
    for name, unit in _METRIC_UNITS.items():
        try:
            metric = summary[name]
            value = _normalize_metric(metric, unit, name)
        except (KeyError, TypeError, ValueError) as exc:
            metrics[name] = None
            diagnostics.append(str(exc))
            continue
        metrics[name] = {"value": value, "unit": unit}
    return {"metrics": metrics, "diagnostics": diagnostics}


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
    diagnostics = list(result.get("diagnostics", []))
    spectrum = config.get("nsb_spectrum")
    if spectrum:
        spectrum_path = Path(spectrum).resolve()
        if spectrum_path.is_file():
            inputs["spectrum"] = {
                "path": spectrum_path.name,
                "sha256": sha256_file(spectrum_path),
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
        "schema_version": 1,
        "status": ("complete" if exit_status == 0 and not diagnostics else "incomplete"),
        "case": case,
        "software": {
            "simtools_version": simtools_version or __version__,
            "model_version": case["model_version"],
        },
        "image": image_identity,
        "inputs": inputs,
        "metrics": result.get("metrics", {}),
        "artifacts": artifact_paths,
        "exit_status": exit_status,
        "diagnostics": diagnostics,
    }


def collect_legacy_results(output_directory, config, image_identity, simtools_version):
    """Create a manifest from outputs produced by a legacy validator command."""
    output = Path(output_directory).resolve()
    summaries = sorted(output.glob("*camera_efficiency_summary*.yml"))
    if len(summaries) != 1:
        raise ValueError(f"Expected one camera-efficiency summary in {output}.")
    summary_path = summaries[0]
    result = read_summary(summary_path)
    effective_config = dict(config)
    for key in ("telescope", "model_version"):
        if result["case"].get(key):
            effective_config[key] = result["case"][key]
    for key in ("zenith_angle", "azimuth_angle"):
        if effective_config.get(key) is None:
            effective_config[key] = result["case"].get(key)
    if not effective_config.get("nsb_spectrum"):
        effective_config["nsb_spectrum"] = result["case"].get("nsb_spectrum")
    effective_config.setdefault("skip_correction_to_nsb_spectrum", False)
    tables = sorted(output.glob("*.ecsv"))
    plots = sorted(
        path for path in output.iterdir() if path.suffix.lower() in {".png", ".pdf", ".svg"}
    )
    artifacts = {"summary": summary_path, "wavelength_tables": tables, "native_plots": plots}
    shower_table = next((path for path in tables if "shower" in path.name.lower()), None)
    if shower_table:
        artifacts["wavelength_table"] = shower_table
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
    changes, metric_diagnostics = _compare_metrics(baseline, candidate, tolerances)
    diagnostics.extend(metric_diagnostics)
    diagnostics.extend(baseline.get("diagnostics", []))
    diagnostics.extend(candidate.get("diagnostics", []))
    if baseline.get("status") != "complete" or candidate.get("status") != "complete":
        diagnostics.append("One or both input manifests are incomplete.")
    return {
        "schema_version": 1,
        "status": "incomplete" if diagnostics else "complete",
        "baseline": baseline.get("case", {}),
        "candidate": candidate.get("case", {}),
        "changes": changes,
        "diagnostics": list(dict.fromkeys(diagnostics)),
        "plots": {},
    }


def _compatibility_diagnostics(baseline, candidate, compare_by):
    """Find undeclared case-setting and input-hash differences."""
    diagnostics = []
    known_paths = {f"case.{field}" for field in _CASE_FIELDS}
    input_names = set(baseline.get("inputs", {})) | set(candidate.get("inputs", {}))
    known_paths.update(f"inputs.{name}.sha256" for name in input_names)
    for path in compare_by - known_paths:
        diagnostics.append(f"Unsupported declared comparison field {path}.")
    for field in _CASE_FIELDS:
        path = f"case.{field}"
        if path in compare_by:
            continue
        if baseline.get("case", {}).get(field) != candidate.get("case", {}).get(field):
            diagnostics.append(f"Incompatible case setting {path}.")
    for name in input_names:
        path = f"inputs.{name}.sha256"
        if path not in compare_by:
            left = baseline.get("inputs", {}).get(name, {}).get("sha256")
            right = candidate.get("inputs", {}).get(name, {}).get("sha256")
            if left != right:
                diagnostics.append(f"Incompatible input hash {path}.")
    return diagnostics


def _compare_metrics(baseline, candidate, tolerances):
    """Compare each declared metric and gather per-metric diagnostics."""
    changes = {}
    diagnostics = []
    baseline_metrics = baseline.get("metrics", {})
    candidate_metrics = candidate.get("metrics", {})
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
        before = _finite_number(left["value"], name)
        after = _finite_number(right["value"], name)
        source_unit = u.Unit(right.get("unit", ""))
        target_unit = u.Unit(left.get("unit", ""))
        after = float((after * source_unit).to_value(target_unit))
    except (KeyError, TypeError, ValueError, u.UnitConversionError) as exc:
        return None, f"Incompatible metric {name}: {exc}"
    delta = after - before
    relative = None if before == 0 else delta / before
    return {
        "baseline": {"value": before, "unit": left.get("unit", "")},
        "candidate": {"value": after, "unit": left.get("unit", "")},
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
    output.write_text(json.dumps(value, indent=2, default=_json_value) + "\n", encoding="utf-8")


def load_manifest(path):
    """Load a manifest or adapt a legacy YAML summary with unknown provenance."""
    result_path = Path(path)
    if result_path.is_dir():
        manifest_path = result_path / "camera_efficiency_manifest.json"
        if manifest_path.is_file():
            result_path = manifest_path
        else:
            summaries = sorted(result_path.glob("*camera_efficiency_summary*.yml"))
            if len(summaries) != 1:
                raise ValueError(f"Expected one camera-efficiency summary in {result_path}.")
            result_path = summaries[0]
    if result_path.suffix.lower() in {".yml", ".yaml"}:
        summary = read_summary(result_path)
        summary.update(
            {
                "schema_version": 1,
                "status": "incomplete",
                "software": {},
                "image": None,
                "inputs": {},
                "artifacts": {"summary": result_path.name},
                "diagnostics": summary["diagnostics"]
                + ["Legacy summary does not contain complete case provenance."],
            }
        )
        summary["_root"] = result_path.parent
        return summary
    manifest = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported camera-efficiency manifest: {result_path}")
    manifest["_root"] = result_path.parent
    return manifest


def _config_value(config, name):
    value = config.get(name)
    if name == "skip_correction_to_nsb_spectrum" and value is None:
        return False
    if hasattr(value, "to_value"):
        value = value.to_value("deg")
    if name == "nsb_spectrum" and value:
        value = Path(value).name
    elif name == "nsb_spectrum" and value is None:
        value = "default sim_telarray spectrum"
    return _plain_value(value)


def _relative_artifact_path(value, root):
    if isinstance(value, list):
        return [_relative_artifact_path(item, root) for item in value]
    return Path(value).resolve().relative_to(root).as_posix()


def _plain_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "value"):
        return _plain_value(value.value)
    return value


def _angle_value(value):
    if isinstance(value, dict) and {"value", "unit"} <= value.keys():
        return (value["value"] * u.Unit(value["unit"])).to_value(u.deg)
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
    value = metric["value"]
    if isinstance(value, u.Quantity):
        return _metric_value(value, expected_unit, name)
    number = _finite_number(value, name)
    unit = u.Unit(metric.get("unit", expected_unit))
    return (number * unit).to_value(expected_unit)


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "to_value"):
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

import json
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
import yaml

from simtools.science_tests import camera_efficiency_results as results

_CAMERA_EFFICIENCY_METRIC = "cam_eff"
_CASE_KEY = "case"
_SOFTWARE_KEY = "software"
_INPUTS_KEY = "inputs"
_METRICS_KEY = "metrics"
_ARTIFACTS_KEY = "artifacts"
_SCHEMA_VERSION_KEY = "schema_version"
_ENCODING = "utf-8"
_DIAGNOSTICS_KEY = "diagnostics"

_DEFAULT_NSB_SPECTRUM = "default sim_telarray spectrum"
_SUMMARY_FILE_NAME = "summary.yml"
_NSB_RATE_METRIC = "nsb_rate"
_METRIC_VALUE_KEY = "value"
_METRIC_UNIT_KEY = "unit"
_GHZ_UNIT = "GHz"


def test_read_summary_normalizes_values_and_units(tmp_test_directory):
    summary = Path(tmp_test_directory) / _SUMMARY_FILE_NAME
    summary.write_text(
        yaml.safe_dump(
            {
                "meta": {"tel": "MSTN-01", "model_version": "test"},
                _CAMERA_EFFICIENCY_METRIC: {_METRIC_VALUE_KEY: 0.5},
                _NSB_RATE_METRIC: {_METRIC_VALUE_KEY: 0.2},
            }
        ),
        encoding=_ENCODING,
    )

    parsed = results.read_summary(summary)

    assert parsed[_METRICS_KEY][_CAMERA_EFFICIENCY_METRIC] == {
        _METRIC_VALUE_KEY: 0.5,
        _METRIC_UNIT_KEY: "",
    }
    assert parsed[_METRICS_KEY][_NSB_RATE_METRIC] == {
        _METRIC_VALUE_KEY: 0.2,
        _METRIC_UNIT_KEY: _GHZ_UNIT,
    }
    assert parsed[_CASE_KEY]["telescope"] == "MSTN-01"


def test_read_summary_normalizes_nested_metric_value(tmp_test_directory):
    summary = Path(tmp_test_directory) / _SUMMARY_FILE_NAME
    summary.write_text(
        yaml.safe_dump(
            {
                _NSB_RATE_METRIC: {
                    _METRIC_VALUE_KEY: {_METRIC_VALUE_KEY: 0.25, _METRIC_UNIT_KEY: _GHZ_UNIT}
                }
            }
        ),
        encoding=_ENCODING,
    )

    parsed = results.read_summary(summary)

    assert parsed[_METRICS_KEY][_NSB_RATE_METRIC] == {
        _METRIC_VALUE_KEY: 0.25,
        _METRIC_UNIT_KEY: _GHZ_UNIT,
    }


def test_read_summary_rejects_non_mapping(tmp_test_directory):
    path = Path(tmp_test_directory) / _SUMMARY_FILE_NAME
    path.write_text("- not a summary\n", encoding=_ENCODING)

    with pytest.raises(ValueError, match="must be a mapping"):
        results.read_summary(path)


def test_normalize_summary_handles_quantities_and_invalid_entries():
    normalized = results.normalize_summary(
        {
            _CAMERA_EFFICIENCY_METRIC: {_METRIC_VALUE_KEY: 50 * u.percent},
            _NSB_RATE_METRIC: {_METRIC_VALUE_KEY: "invalid"},
        }
    )

    assert normalized[_METRICS_KEY][_CAMERA_EFFICIENCY_METRIC] == {
        _METRIC_VALUE_KEY: 0.5,
        _METRIC_UNIT_KEY: "",
    }
    assert normalized[_METRICS_KEY][_NSB_RATE_METRIC] is None
    assert len(normalized[_DIAGNOSTICS_KEY]) == 6


@pytest.mark.parametrize(_METRIC_VALUE_KEY, [float("nan"), float("inf"), "bad", True])
def test_read_summary_reports_non_finite_or_non_numeric_metric(tmp_test_directory, value):
    summary = Path(tmp_test_directory) / _SUMMARY_FILE_NAME
    summary.write_text(
        yaml.safe_dump({_CAMERA_EFFICIENCY_METRIC: {_METRIC_VALUE_KEY: value}}), encoding=_ENCODING
    )

    parsed = results.read_summary(summary)

    assert parsed[_METRICS_KEY][_CAMERA_EFFICIENCY_METRIC] is None
    assert any(_CAMERA_EFFICIENCY_METRIC in item for item in parsed[_DIAGNOSTICS_KEY])


def test_manifest_hashes_explicit_spectrum_and_records_relative_artifacts(tmp_test_directory):
    root = Path(tmp_test_directory)
    spectrum = root / "spectrum.dat"
    spectrum.write_text("350 1\n", encoding=_ENCODING)
    manifest = results.create_manifest(
        {_METRICS_KEY: {_CAMERA_EFFICIENCY_METRIC: {_METRIC_VALUE_KEY: 0.5, _METRIC_UNIT_KEY: ""}}},
        {
            "site": "North",
            "telescope": "MSTN-01",
            "model_version": "1.2",
            "zenith_angle": 20 * u.deg,
            "azimuth_angle": 0 * u.deg,
            "nsb_spectrum": str(spectrum),
            "skip_correction_to_nsb_spectrum": False,
        },
        root,
        artifacts={"summary": root / _SUMMARY_FILE_NAME},
        image_identity="sha256:abc",
    )

    assert manifest[_INPUTS_KEY]["spectrum"]["sha256"] == results.sha256_file(spectrum)
    assert manifest[_ARTIFACTS_KEY] == {"summary": _SUMMARY_FILE_NAME}
    assert manifest["image"] == "sha256:abc"


def test_manifest_marks_missing_provenance_and_failed_exit_incomplete(tmp_test_directory):
    manifest = results.create_manifest(
        {_METRICS_KEY: {}, _DIAGNOSTICS_KEY: ["missing output"]},
        {"site": "North"},
        tmp_test_directory,
        exit_status=1,
    )

    assert manifest["status"] == "incomplete"
    assert manifest[_SOFTWARE_KEY]["simtools_version"]
    assert len(manifest[_DIAGNOSTICS_KEY]) == 3


def test_manifest_does_not_hash_default_spectrum_label(tmp_test_directory):
    manifest = results.create_manifest(
        {_METRICS_KEY: {}},
        {"nsb_spectrum": _DEFAULT_NSB_SPECTRUM},
        tmp_test_directory,
        image_identity="image",
        simtools_version="0.39.0",
    )

    assert manifest[_INPUTS_KEY] == {}
    assert manifest[_DIAGNOSTICS_KEY] == []


def test_collect_legacy_results_fills_case_and_indexes_products(tmp_test_directory):
    from astropy.table import Table

    root = Path(tmp_test_directory)
    summary_values = {
        name: {_METRIC_VALUE_KEY: 0.5 if unit == "" else 1.0, _METRIC_UNIT_KEY: unit}
        for name, unit in results._METRIC_UNITS.items()
    }
    summary_values["meta"] = {
        "tel": "MSTN-01",
        "model_version": "7.0.0",
        "zen": {_METRIC_VALUE_KEY: 10, _METRIC_UNIT_KEY: "deg"},
        "az": {_METRIC_VALUE_KEY: 20, _METRIC_UNIT_KEY: "deg"},
        "nsb": str(root / "spectrum.ecsv"),
    }
    (root / "spectrum.ecsv").write_text("350 1\n", encoding=_ENCODING)
    (root / "camera_efficiency_summary.yml").write_text(
        yaml.safe_dump(summary_values), encoding=_ENCODING
    )
    Table({"wl": [300], "eff": [0.5]}).write(root / "shower.ecsv", format="ascii.ecsv")
    (root / "native.png").write_bytes(b"plot")

    manifest = results.collect_legacy_results(root, {"site": "North"}, "image", "0.38.1")

    assert manifest["status"] == "complete"
    assert manifest[_CASE_KEY]["telescope"] == "MSTN-01"
    assert manifest[_CASE_KEY]["nsb_spectrum"] == "spectrum.ecsv"
    assert manifest[_ARTIFACTS_KEY]["wavelength_table"] == "shower.ecsv"
    assert manifest[_ARTIFACTS_KEY]["native_plots"] == ["native.png"]


@pytest.mark.parametrize("count", [0, 2])
def test_collect_legacy_results_requires_one_summary(tmp_test_directory, count):
    root = Path(tmp_test_directory)
    for index in range(count):
        (root / f"camera_efficiency_summary_{index}.yml").write_text("{}", encoding=_ENCODING)

    with pytest.raises(ValueError, match="Expected one"):
        results.collect_legacy_results(root, {}, "image", "0.38.1")


def test_compare_manifests_checks_declared_changes_and_zero_baseline():
    baseline = _manifest(0.0, model_version="1.0")
    candidate = _manifest(0.2, model_version="2.0")

    comparison = results.compare_manifests(baseline, candidate, compare_by=["case.model_version"])

    assert comparison["status"] == "complete"
    assert comparison["baseline"][_SOFTWARE_KEY]["simtools_version"] == "test"
    assert comparison["baseline"]["image"] == "image"
    assert comparison["changes"][_CAMERA_EFFICIENCY_METRIC]["relative"] is None
    assert comparison["changes"][_CAMERA_EFFICIENCY_METRIC]["relative_reason"] == "baseline is zero"


def test_compare_manifests_marks_unapproved_case_difference_incomplete():
    comparison = results.compare_manifests(_manifest(0.2), _manifest(0.3, site="South"))

    assert comparison["status"] == "incomplete"
    assert any("case.site" in item for item in comparison[_DIAGNOSTICS_KEY])


def test_compare_manifests_rejects_missing_provenance_on_complete_manifest():
    baseline = _manifest(0.2)
    candidate = _manifest(0.3)
    candidate[_CASE_KEY].pop("telescope")
    candidate["image"] = None

    comparison = results.compare_manifests(baseline, candidate)

    assert comparison["status"] == "incomplete"
    assert any(
        "Missing candidate case provenance: telescope" in item
        for item in comparison[_DIAGNOSTICS_KEY]
    )
    assert any("Missing candidate image identity" in item for item in comparison[_DIAGNOSTICS_KEY])


def test_compare_manifests_rejects_unknown_declared_field():
    comparison = results.compare_manifests(
        _manifest(0.2), _manifest(0.3), compare_by=["case.unrecognized"]
    )

    assert comparison["status"] == "incomplete"
    assert any(
        "Unsupported declared comparison field" in item for item in comparison[_DIAGNOSTICS_KEY]
    )


def test_compare_manifests_checks_hashes_provenance_and_advisory_tolerance():
    baseline = _manifest(1.0)
    candidate = _manifest(1.2)
    baseline[_INPUTS_KEY] = {"spectrum": {"sha256": "before"}}
    candidate[_INPUTS_KEY] = {"spectrum": {"sha256": "after"}}
    candidate["status"] = "incomplete"

    comparison = results.compare_manifests(
        baseline, candidate, tolerances={_CAMERA_EFFICIENCY_METRIC: 0.05}
    )
    declared = results.compare_manifests(
        baseline,
        candidate,
        compare_by=["inputs.spectrum.sha256"],
        tolerances={_CAMERA_EFFICIENCY_METRIC: 0.05},
    )

    assert any("input hash" in item for item in comparison[_DIAGNOSTICS_KEY])
    assert any("manifests are incomplete" in item for item in comparison[_DIAGNOSTICS_KEY])
    assert comparison["changes"][_CAMERA_EFFICIENCY_METRIC]["advisory"] is True
    assert not any("input hash" in item for item in declared[_DIAGNOSTICS_KEY])


def test_compare_manifests_converts_units_and_marks_bad_metrics_incomplete():
    baseline = _manifest(1.0)
    candidate = _manifest(1000.0)
    baseline[_METRICS_KEY][_NSB_RATE_METRIC] = {_METRIC_VALUE_KEY: 1.0, _METRIC_UNIT_KEY: _GHZ_UNIT}
    candidate[_METRICS_KEY][_NSB_RATE_METRIC] = {_METRIC_VALUE_KEY: 1000.0, _METRIC_UNIT_KEY: "MHz"}
    candidate[_METRICS_KEY][_CAMERA_EFFICIENCY_METRIC] = None

    comparison = results.compare_manifests(baseline, candidate)

    assert comparison["changes"][_NSB_RATE_METRIC]["absolute"] == pytest.approx(0.0)
    assert comparison["status"] == "incomplete"


def test_json_writer(tmp_test_directory):
    path = Path(tmp_test_directory) / "result.json"
    results.write_json(path, {_METRIC_VALUE_KEY: np.float64(1.0)})
    assert json.loads(path.read_text(encoding=_ENCODING)) == {_METRIC_VALUE_KEY: 1.0}


def test_load_manifest_adapts_legacy_directory_and_checks_schema(tmp_test_directory):
    root = Path(tmp_test_directory)
    summary = {name: {_METRIC_VALUE_KEY: 0.5} for name in results._METRIC_UNITS}
    (root / "camera_efficiency_summary.yml").write_text(yaml.safe_dump(summary))

    legacy = results.load_manifest(root)
    assert legacy["status"] == "incomplete"
    assert legacy["_root"] == root

    manifest_path = root / "camera_efficiency_manifest.json"
    manifest_path.write_text(json.dumps({_SCHEMA_VERSION_KEY: 99}))
    with pytest.raises(ValueError, match="Unsupported"):
        results.load_manifest(root)


def test_load_manifest_reads_supported_json_manifest(tmp_test_directory):
    root = Path(tmp_test_directory)
    path = root / "camera_efficiency_manifest.json"
    path.write_text(json.dumps({_SCHEMA_VERSION_KEY: 1, "status": "complete"}))

    manifest = results.load_manifest(root)

    assert manifest["status"] == "complete"
    assert manifest["_root"] == root


def test_json_writer_rejects_unrecognized_objects(tmp_test_directory):
    path = Path(tmp_test_directory) / "result.json"

    with pytest.raises(TypeError, match="not JSON serializable"):
        results.write_json(path, {_METRIC_VALUE_KEY: object()})


def test_config_and_angle_helpers_normalize_optional_values():
    assert results._config_value({"nsb_spectrum": None}, "nsb_spectrum") == (_DEFAULT_NSB_SPECTRUM)
    assert (
        results._config_value(
            {"skip_correction_to_nsb_spectrum": None}, "skip_correction_to_nsb_spectrum"
        )
        is False
    )
    assert results._angle_value(20 * u.deg) == 20
    assert results._plain_value(np.int64(2)) == 2
    assert results._metric_value(50 * u.percent, "", _CAMERA_EFFICIENCY_METRIC) == pytest.approx(
        0.5
    )


def _manifest(value, site="North", model_version="1.0"):
    metrics = {
        name: {_METRIC_VALUE_KEY: 0.5, _METRIC_UNIT_KEY: ""} for name in results._METRIC_UNITS
    }
    metrics[_CAMERA_EFFICIENCY_METRIC] = {_METRIC_VALUE_KEY: value, _METRIC_UNIT_KEY: ""}
    metrics[_NSB_RATE_METRIC] = {_METRIC_VALUE_KEY: 1.0, _METRIC_UNIT_KEY: _GHZ_UNIT}
    return {
        _SCHEMA_VERSION_KEY: 1,
        "status": "complete",
        _CASE_KEY: {
            "site": site,
            "telescope": "MSTN-01",
            "zenith_angle": 0.0,
            "azimuth_angle": 0.0,
            "nsb_spectrum": _DEFAULT_NSB_SPECTRUM,
            "skip_correction_to_nsb_spectrum": False,
            "model_version": model_version,
        },
        _SOFTWARE_KEY: {"simtools_version": "test"},
        "image": "image",
        _INPUTS_KEY: {},
        _METRICS_KEY: metrics,
        _ARTIFACTS_KEY: {},
    }


@pytest.mark.parametrize(
    "field", [_CASE_KEY, _SOFTWARE_KEY, _INPUTS_KEY, _METRICS_KEY, _ARTIFACTS_KEY]
)
def test_manifest_loader_rejects_malformed_container_fields(tmp_test_directory, field):
    path = Path(tmp_test_directory) / "manifest.json"
    path.write_text(json.dumps({_SCHEMA_VERSION_KEY: 1, field: []}), encoding=_ENCODING)

    with pytest.raises(ValueError, match=f"Manifest field {field} must be a mapping"):
        results.load_manifest(path)


def test_metric_comparison_rejects_overflowing_changes():
    change, diagnostic = results._metric_change(
        _CAMERA_EFFICIENCY_METRIC, {"value": 1e308}, {"value": -1e308}, None
    )

    assert change is None
    assert "Non-finite change" in diagnostic

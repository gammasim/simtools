import json
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
import yaml

from simtools.science_tests import camera_efficiency_results as results


def test_read_summary_normalizes_values_and_units(tmp_test_directory):
    summary = Path(tmp_test_directory) / "summary.yml"
    summary.write_text(
        yaml.safe_dump(
            {
                "meta": {"tel": "MSTN-01", "model_version": "test"},
                "cam_eff": {"value": 0.5},
                "nsb_rate": {"value": 0.2},
            }
        ),
        encoding="utf-8",
    )

    parsed = results.read_summary(summary)

    assert parsed["metrics"]["cam_eff"] == {"value": 0.5, "unit": ""}
    assert parsed["metrics"]["nsb_rate"] == {"value": 0.2, "unit": "GHz"}
    assert parsed["case"]["telescope"] == "MSTN-01"


def test_read_summary_rejects_non_mapping(tmp_test_directory):
    path = Path(tmp_test_directory) / "summary.yml"
    path.write_text("- not a summary\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a mapping"):
        results.read_summary(path)


def test_normalize_summary_handles_quantities_and_invalid_entries():
    normalized = results.normalize_summary(
        {"cam_eff": {"value": 50 * u.percent}, "nsb_rate": {"value": "invalid"}}
    )

    assert normalized["metrics"]["cam_eff"] == {"value": 0.5, "unit": ""}
    assert normalized["metrics"]["nsb_rate"] is None
    assert len(normalized["diagnostics"]) == 6


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "bad", True])
def test_read_summary_reports_non_finite_or_non_numeric_metric(tmp_test_directory, value):
    summary = Path(tmp_test_directory) / "summary.yml"
    summary.write_text(yaml.safe_dump({"cam_eff": {"value": value}}), encoding="utf-8")

    parsed = results.read_summary(summary)

    assert parsed["metrics"]["cam_eff"] is None
    assert any("cam_eff" in item for item in parsed["diagnostics"])


def test_manifest_hashes_explicit_spectrum_and_records_relative_artifacts(tmp_test_directory):
    root = Path(tmp_test_directory)
    spectrum = root / "spectrum.dat"
    spectrum.write_text("350 1\n", encoding="utf-8")
    manifest = results.create_manifest(
        {"metrics": {"cam_eff": {"value": 0.5, "unit": ""}}},
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
        artifacts={"summary": root / "summary.yml"},
        image_identity="sha256:abc",
    )

    assert manifest["inputs"]["spectrum"]["sha256"] == results.sha256_file(spectrum)
    assert manifest["artifacts"] == {"summary": "summary.yml"}
    assert manifest["image"] == "sha256:abc"


def test_manifest_marks_missing_provenance_and_failed_exit_incomplete(tmp_test_directory):
    manifest = results.create_manifest(
        {"metrics": {}, "diagnostics": ["missing output"]},
        {"site": "North"},
        tmp_test_directory,
        exit_status=1,
    )

    assert manifest["status"] == "incomplete"
    assert manifest["software"]["simtools_version"]
    assert len(manifest["diagnostics"]) == 3


def test_collect_legacy_results_fills_case_and_indexes_products(tmp_test_directory):
    from astropy.table import Table

    root = Path(tmp_test_directory)
    summary_values = {
        name: {"value": 0.5 if unit == "" else 1.0, "unit": unit}
        for name, unit in results._METRIC_UNITS.items()
    }
    summary_values["meta"] = {
        "tel": "MSTN-01",
        "model_version": "7.0.0",
        "zen": {"value": 10, "unit": "deg"},
        "az": {"value": 20, "unit": "deg"},
        "nsb": str(root / "spectrum.ecsv"),
    }
    (root / "spectrum.ecsv").write_text("350 1\n", encoding="utf-8")
    (root / "camera_efficiency_summary.yml").write_text(
        yaml.safe_dump(summary_values), encoding="utf-8"
    )
    Table({"wl": [300], "eff": [0.5]}).write(root / "shower.ecsv", format="ascii.ecsv")
    (root / "native.png").write_bytes(b"plot")

    manifest = results.collect_legacy_results(root, {"site": "North"}, "image", "0.38.1")

    assert manifest["status"] == "complete"
    assert manifest["case"]["telescope"] == "MSTN-01"
    assert manifest["case"]["nsb_spectrum"] == "spectrum.ecsv"
    assert manifest["artifacts"]["wavelength_table"] == "shower.ecsv"
    assert manifest["artifacts"]["native_plots"] == ["native.png"]


@pytest.mark.parametrize("count", [0, 2])
def test_collect_legacy_results_requires_one_summary(tmp_test_directory, count):
    root = Path(tmp_test_directory)
    for index in range(count):
        (root / f"camera_efficiency_summary_{index}.yml").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected one"):
        results.collect_legacy_results(root, {}, "image", "0.38.1")


def test_compare_manifests_checks_declared_changes_and_zero_baseline():
    baseline = _manifest(0.0, model_version="1.0")
    candidate = _manifest(0.2, model_version="2.0")

    comparison = results.compare_manifests(baseline, candidate, compare_by=["case.model_version"])

    assert comparison["status"] == "complete"
    assert comparison["changes"]["cam_eff"]["relative"] is None
    assert comparison["changes"]["cam_eff"]["relative_reason"] == "baseline is zero"


def test_compare_manifests_marks_unapproved_case_difference_incomplete():
    comparison = results.compare_manifests(_manifest(0.2), _manifest(0.3, site="South"))

    assert comparison["status"] == "incomplete"
    assert any("case.site" in item for item in comparison["diagnostics"])


def test_compare_manifests_rejects_unknown_declared_field():
    comparison = results.compare_manifests(
        _manifest(0.2), _manifest(0.3), compare_by=["case.unrecognized"]
    )

    assert comparison["status"] == "incomplete"
    assert any(
        "Unsupported declared comparison field" in item for item in comparison["diagnostics"]
    )


def test_compare_manifests_checks_hashes_provenance_and_advisory_tolerance():
    baseline = _manifest(1.0)
    candidate = _manifest(1.2)
    baseline["inputs"] = {"spectrum": {"sha256": "before"}}
    candidate["inputs"] = {"spectrum": {"sha256": "after"}}
    candidate["status"] = "incomplete"

    comparison = results.compare_manifests(baseline, candidate, tolerances={"cam_eff": 0.05})
    declared = results.compare_manifests(
        baseline,
        candidate,
        compare_by=["inputs.spectrum.sha256"],
        tolerances={"cam_eff": 0.05},
    )

    assert any("input hash" in item for item in comparison["diagnostics"])
    assert any("manifests are incomplete" in item for item in comparison["diagnostics"])
    assert comparison["changes"]["cam_eff"]["advisory"] is True
    assert not any("input hash" in item for item in declared["diagnostics"])


def test_compare_manifests_converts_units_and_marks_bad_metrics_incomplete():
    baseline = _manifest(1.0)
    candidate = _manifest(1000.0)
    baseline["metrics"]["nsb_rate"] = {"value": 1.0, "unit": "GHz"}
    candidate["metrics"]["nsb_rate"] = {"value": 1000.0, "unit": "MHz"}
    candidate["metrics"]["cam_eff"] = None

    comparison = results.compare_manifests(baseline, candidate)

    assert comparison["changes"]["nsb_rate"]["absolute"] == pytest.approx(0.0)
    assert comparison["status"] == "incomplete"


def test_json_writer(tmp_test_directory):
    path = Path(tmp_test_directory) / "result.json"
    results.write_json(path, {"value": np.float64(1.0)})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1.0}


def test_load_manifest_adapts_legacy_directory_and_checks_schema(tmp_test_directory):
    root = Path(tmp_test_directory)
    summary = {name: {"value": 0.5} for name in results._METRIC_UNITS}
    (root / "camera_efficiency_summary.yml").write_text(yaml.safe_dump(summary))

    legacy = results.load_manifest(root)
    assert legacy["status"] == "incomplete"
    assert legacy["_root"] == root

    manifest_path = root / "camera_efficiency_manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(ValueError, match="Unsupported"):
        results.load_manifest(root)


def test_load_manifest_reads_supported_json_manifest(tmp_test_directory):
    root = Path(tmp_test_directory)
    path = root / "camera_efficiency_manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "status": "complete"}))

    manifest = results.load_manifest(root)

    assert manifest["status"] == "complete"
    assert manifest["_root"] == root


def test_json_writer_rejects_unrecognized_objects(tmp_test_directory):
    path = Path(tmp_test_directory) / "result.json"

    with pytest.raises(TypeError, match="not JSON serializable"):
        results.write_json(path, {"value": object()})


def test_config_and_angle_helpers_normalize_optional_values():
    assert results._config_value({"nsb_spectrum": None}, "nsb_spectrum") == (
        "default sim_telarray spectrum"
    )
    assert (
        results._config_value(
            {"skip_correction_to_nsb_spectrum": None}, "skip_correction_to_nsb_spectrum"
        )
        is False
    )
    assert results._angle_value(20 * u.deg) == 20
    assert results._plain_value(np.int64(2)) == 2
    assert results._metric_value(50 * u.percent, "", "cam_eff") == pytest.approx(0.5)


def _manifest(value, site="North", model_version="1.0"):
    metrics = {name: {"value": 0.5, "unit": ""} for name in results._METRIC_UNITS}
    metrics["cam_eff"] = {"value": value, "unit": ""}
    metrics["nsb_rate"] = {"value": 1.0, "unit": "GHz"}
    return {
        "schema_version": 1,
        "status": "complete",
        "case": {"site": site, "model_version": model_version},
        "software": {"simtools_version": "test"},
        "image": "image",
        "inputs": {},
        "metrics": metrics,
        "artifacts": {},
    }

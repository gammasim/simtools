import json
from pathlib import Path

import pytest
import yaml

from simtools.applications import compare_science_results as application
from simtools.science_tests.camera_efficiency_results import load_manifest


def test_tolerance_parser_rejects_invalid_or_non_finite_values():
    assert application._parse_tolerances(["cam_eff=0.1"]) == {"cam_eff": 0.1}
    for value in ("unknown=0.1", "cam_eff=-1", "cam_eff=nan"):
        with pytest.raises(ValueError, match=r"Invalid metric|finite"):
            application._parse_tolerances([value])


def test_load_manifest_adapts_legacy_summary_and_marks_unknown_provenance(tmp_test_directory):
    summary = Path(tmp_test_directory) / "camera_efficiency_summary.yml"
    summary.write_text(
        yaml.safe_dump({"meta": {"tel": "MSTN-01"}, "cam_eff": {"value": 0.5}}),
        encoding="utf-8",
    )

    manifest = load_manifest(summary)

    assert manifest["status"] == "incomplete"
    assert manifest["_root"] == summary.parent
    assert any("provenance" in item for item in manifest["diagnostics"])


def test_manifest_loader_requires_supported_version(tmp_test_directory):
    path = Path(tmp_test_directory) / "result.json"
    path.write_text(json.dumps({"schema_version": 3}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported"):
        load_manifest(path)


def test_required_wavelength_artifact_missing_marks_comparison_incomplete():
    comparison = {"status": "complete", "diagnostics": [], "plots": {}}

    application._plot_wavelengths({}, {}, comparison, Path(), True)

    assert comparison["status"] == "incomplete"
    assert any("Required wavelength" in item for item in comparison["diagnostics"])

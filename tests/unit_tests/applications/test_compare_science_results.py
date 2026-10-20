import json
import os
from pathlib import Path

import pytest
import yaml

from simtools.applications import compare_science_results as application
from simtools.science_tests.camera_efficiency_results import load_manifest

_CHANNEL_KEY = "channel"
_SUMMARY_FILE = "camera_efficiency_summary.yml"
_MANIFEST_FILE = "camera_efficiency_manifest.json"
_FIRST_TELESCOPE = "LSTN-01"
_SECOND_TELESCOPE = "LSTN-02"
_STATUS_KEY = "status"
_COMPLETE_STATUS = "complete"
_INCOMPLETE_STATUS = "incomplete"
_CHANGES_KEY = "changes"
_DIAGNOSTICS_KEY = "diagnostics"
_PLOTS_KEY = "plots"
_CAMERA_EFFICIENCY_METRIC = "cam_eff"
_ENCODING = "utf-8"
_ROOT_KEY = "_root"
_TELESCOPES_KEY = "telescopes"
_TELESCOPE_KEY = "telescope"
_VALUE_KEY = "value"
_BASELINE_LABEL_KEY = "baseline_label"
_CANDIDATE_LABEL_KEY = "candidate_label"
_REQUIRE_TABLES_KEY = "require_wavelength_tables"
_INVALID_TABLE_ERROR = "Invalid ECSV header"


def test_tolerance_parser_rejects_invalid_or_non_finite_values():
    assert application._parse_tolerances(["cam_eff=0.1"]) == {_CAMERA_EFFICIENCY_METRIC: 0.1}
    for value in ("unknown=0.1", "cam_eff=-1", "cam_eff=nan"):
        with pytest.raises(ValueError, match=r"Invalid metric|finite"):
            application._parse_tolerances([value])


def test_tolerance_parser_accepts_values_grouped_by_one_option():
    assert application._parse_tolerances(
        [["reflectivity=0.05", "cam_eff=0.05", "tel_eff=0.05"]]
    ) == {"reflectivity": 0.05, _CAMERA_EFFICIENCY_METRIC: 0.05, "tel_eff": 0.05}


def test_load_manifest_adapts_legacy_summary_and_marks_unknown_provenance(tmp_test_directory):
    summary = Path(tmp_test_directory) / _SUMMARY_FILE
    summary.write_text(
        yaml.safe_dump({"meta": {"tel": "MSTN-01"}, _CAMERA_EFFICIENCY_METRIC: {_VALUE_KEY: 0.5}}),
        encoding=_ENCODING,
    )

    manifest = load_manifest(summary)

    assert manifest[_STATUS_KEY] == _INCOMPLETE_STATUS
    assert manifest[_ROOT_KEY] == summary.parent
    assert any("provenance" in item for item in manifest[_DIAGNOSTICS_KEY])


def test_manifest_loader_requires_supported_version(tmp_test_directory):
    path = Path(tmp_test_directory) / "result.json"
    path.write_text(json.dumps({"schema_version": 3}), encoding=_ENCODING)

    with pytest.raises(ValueError, match="Unsupported"):
        load_manifest(path)


@pytest.mark.parametrize("newest", ["root", "nested"])
def test_manifest_set_uses_most_recent_telescope_output(tmp_test_directory, monkeypatch, newest):
    root = Path(tmp_test_directory)
    direct = root / _MANIFEST_FILE
    nested = root / _SECOND_TELESCOPE / _MANIFEST_FILE
    nested.parent.mkdir()
    direct.touch()
    nested.touch()
    most_recent = direct if newest == "root" else nested
    os.utime(direct, ns=(1, 1))
    os.utime(nested, ns=(1, 1))
    os.utime(most_recent, ns=(2, 2))
    monkeypatch.setattr(
        application,
        "load_manifest",
        lambda path: {"case": {_TELESCOPE_KEY: _SECOND_TELESCOPE}, _ROOT_KEY: Path(path).parent},
    )

    manifests = application._load_manifest_set(root)

    assert list(manifests) == [_SECOND_TELESCOPE]
    assert manifests[_SECOND_TELESCOPE][_ROOT_KEY] == most_recent.parent


def test_compare_manifest_sets_writes_combined_report_for_selected_telescopes(
    tmp_test_directory, monkeypatch
):
    identity = {"case": {_TELESCOPE_KEY: _FIRST_TELESCOPE}, "software": {}}
    monkeypatch.setattr(
        application,
        "compare_manifests",
        lambda *_args: {
            _STATUS_KEY: _COMPLETE_STATUS,
            "baseline": {},
            "candidate": {},
            _CHANGES_KEY: {_CAMERA_EFFICIENCY_METRIC: {}},
            _DIAGNOSTICS_KEY: [],
            _PLOTS_KEY: {},
        },
    )
    monkeypatch.setattr(application, "_plot_tables", lambda *_args: None)
    plot_calls = []

    def record_plot(*_args, **kwargs):
        plot_calls.append(kwargs)
        return {}, []

    monkeypatch.setattr(application, "plot_comparison", record_plot)
    manifests = {_FIRST_TELESCOPE: identity, _SECOND_TELESCOPE: identity}
    args = {
        _TELESCOPES_KEY: [_SECOND_TELESCOPE, _FIRST_TELESCOPE],
        _BASELINE_LABEL_KEY: "v0.37.0",
        _CANDIDATE_LABEL_KEY: "v0.39.0",
    }

    result = application._compare_manifest_sets(
        manifests, manifests, args, {}, Path(tmp_test_directory)
    )

    assert result[_STATUS_KEY] == _COMPLETE_STATUS
    assert list(result["comparisons"]) == [_SECOND_TELESCOPE, _FIRST_TELESCOPE]
    assert result[_PLOTS_KEY]["report"].endswith("report.pdf")
    assert all(call["figure_format"] == () for call in plot_calls)


def test_compare_manifest_sets_reports_missing_selected_telescope(tmp_test_directory):
    with pytest.raises(ValueError, match="Baseline and candidate telescope sets differ"):
        application._compare_manifest_sets(
            {_FIRST_TELESCOPE: {}},
            {},
            {_TELESCOPES_KEY: [_FIRST_TELESCOPE]},
            {},
            tmp_test_directory,
        )


def test_required_wavelength_artifact_missing_marks_comparison_incomplete():
    comparison = {_STATUS_KEY: _COMPLETE_STATUS, _DIAGNOSTICS_KEY: [], _PLOTS_KEY: {}}

    application._plot_tables({}, {}, comparison, Path(), {_REQUIRE_TABLES_KEY: True}, None)

    assert comparison[_STATUS_KEY] == _INCOMPLETE_STATUS
    assert any("Required wavelength" in item for item in comparison[_DIAGNOSTICS_KEY])


def test_manifest_set_loads_legacy_directory(tmp_test_directory):
    root = Path(tmp_test_directory)
    (root / _SUMMARY_FILE).write_text(
        yaml.safe_dump(
            {"meta": {"tel": _SECOND_TELESCOPE}, _CAMERA_EFFICIENCY_METRIC: {_VALUE_KEY: 0.5}}
        ),
        encoding=_ENCODING,
    )

    manifests = application._load_manifest_set(root)

    assert manifests[_SECOND_TELESCOPE][_STATUS_KEY] == _INCOMPLETE_STATUS


def test_plot_tables_preserves_metrics_when_tables_cannot_be_read(monkeypatch):
    comparison = {
        _STATUS_KEY: _COMPLETE_STATUS,
        _CHANGES_KEY: {_CAMERA_EFFICIENCY_METRIC: {}},
        _DIAGNOSTICS_KEY: [],
    }
    monkeypatch.setattr(application, "_artifact_path", lambda *_args: Path("table.ecsv"))

    def invalid_table(*_args, **_kwargs):
        raise ValueError(_INVALID_TABLE_ERROR)

    monkeypatch.setattr(application, "plot_wavelength_comparisons", invalid_table)
    monkeypatch.setattr(application, "_plot_efficiency_components_for_channel", invalid_table)
    application._plot_tables(
        {},
        {},
        comparison,
        Path(),
        {_REQUIRE_TABLES_KEY: True, _BASELINE_LABEL_KEY: "old", _CANDIDATE_LABEL_KEY: "new"},
        None,
    )

    assert comparison[_STATUS_KEY] == _INCOMPLETE_STATUS
    assert _CAMERA_EFFICIENCY_METRIC in comparison[_CHANGES_KEY]
    assert all(_INVALID_TABLE_ERROR in diagnostic for diagnostic in comparison[_DIAGNOSTICS_KEY])


def test_multiple_telescopes_produce_a_pdf_accepted_by_campaign(tmp_test_directory):
    from astropy.table import Table

    from simtools.science_tests.camera_efficiency_results import create_manifest, normalize_summary
    from simtools.science_tests.definitions import _comparison_status_warnings

    output = Path(tmp_test_directory)
    columns = {
        "wl": [300, 400],
        "ref": [0.8, 0.9],
        "masts": [0.9, 0.9],
        "filt": [0.8, 0.8],
        "pixel": [0.9, 0.9],
        "qe": [0.4, 0.5],
        "atm_trans": [0.8, 0.9],
        "eff": [0.2, 0.3],
        "eff_atm": [0.16, 0.27],
        "cherenkov_detected": [0.3, 0.4],
        "nsb_detected": [10, 20],
    }
    metric_names = (
        "reflectivity",
        _CAMERA_EFFICIENCY_METRIC,
        "tel_eff",
        "tot_sens",
        "nsb_rate",
        "nsb_ref",
        "muon_frac",
    )
    manifests = []
    for label, value in (("reference", 1.0), ("candidate", 1.1)):
        for telescope in (_FIRST_TELESCOPE, _SECOND_TELESCOPE):
            root = output / label / telescope
            root.mkdir(parents=True)
            table_path = root / "table.ecsv"
            Table(columns).write(table_path, format="ascii.ecsv")
            manifest = create_manifest(
                normalize_summary({name: {_VALUE_KEY: value} for name in metric_names}),
                {
                    "site": "North",
                    _TELESCOPE_KEY: telescope,
                    "model_version": "7.0.0",
                    "zenith_angle": 0,
                    "azimuth_angle": 0,
                },
                root,
                artifacts={
                    f"{channel}_wavelength_table": table_path
                    for channel in ("shower", "muon", "nsb")
                },
                image_identity=f"sha256:{label}",
                simtools_version="1.0.0",
            )
            application.write_json(root / _MANIFEST_FILE, manifest)
        manifests.append(application._load_manifest_set(output / label))
    args = {
        _BASELINE_LABEL_KEY: "reference",
        _CANDIDATE_LABEL_KEY: "candidate",
        _REQUIRE_TABLES_KEY: True,
    }

    result = application._compare_manifest_sets(
        *manifests, args, {_CAMERA_EFFICIENCY_METRIC: 0.05}, output
    )
    status_path = output / "comparison.json"
    application.write_json(status_path, result)
    warnings = _comparison_status_warnings({"result_status_file": str(status_path)}, {})

    assert result[_STATUS_KEY] == _COMPLETE_STATUS
    assert len(warnings) == 2
    assert all("cam_eff" in warning for warning in warnings)
    assert (output / "report.pdf").is_file()
    assert not list(output.rglob("*.png"))


def test_plot_tables_adds_component_pages_for_each_channel(monkeypatch):
    comparison = {_STATUS_KEY: _COMPLETE_STATUS, _DIAGNOSTICS_KEY: []}
    monkeypatch.setattr(application, "_artifact_path", lambda _manifest, name: Path(name))
    monkeypatch.setattr(application, "plot_wavelength_comparisons", lambda *_a, **_k: (None, None))
    calls = []

    def capture_components(*args, **kwargs):
        calls.append((args, kwargs))
        return None, None

    monkeypatch.setattr(application, "_plot_efficiency_components_for_channel", capture_components)
    application._plot_tables(
        {"case": {"telescope": _FIRST_TELESCOPE}},
        {},
        comparison,
        Path(),
        {_REQUIRE_TABLES_KEY: True, _BASELINE_LABEL_KEY: "old", _CANDIDATE_LABEL_KEY: "new"},
        None,
    )

    assert [kwargs[_CHANNEL_KEY] for _, kwargs in calls] == ["shower", "muon", "nsb"]
    for args, kwargs in calls:
        assert kwargs[_CHANNEL_KEY] in str(args[0])
        assert kwargs[_CHANNEL_KEY] in str(args[1])
        assert kwargs["telescope"] == _FIRST_TELESCOPE
    assert comparison[_DIAGNOSTICS_KEY] == []

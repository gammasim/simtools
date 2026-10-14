import json
import logging
from pathlib import Path

import yaml

from simtools.science_tests import reports as science_test_reports

_RELEASE = "release"
_QUALIFIED = "qualified"
_RESULTS = "results"
_PRODUCTION_HISTORY = "production_history"
_STATUS = "status"
_EXECUTION = "execution"


_DEPENDENCY_COMPATIBILITY = "dependency_compatibility"


def _write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_production_history_survives_a_non_production_configuration_change(campaign):
    catalogue_path = campaign["template_dir"] / "catalogue.yml"
    catalogue = yaml.safe_load(catalogue_path.read_text(encoding="utf-8"))
    catalogue["tests"]["production"] = {"affects_production": True}
    _write_yaml(catalogue_path, catalogue)
    report_path = campaign["release_dir"] / "reports/release-summary.json"
    report_path.parent.mkdir()
    report_path.write_text(
        json.dumps(
            {
                "campaign_sha256": "old",
                _RESULTS: [],
                _PRODUCTION_HISTORY: [
                    {
                        "id": "production.north",
                        "test": "production",
                        "site": "north",
                        _EXECUTION: "submitted",
                        _STATUS: "submitted",
                        "run_id": "run-1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = science_test_reports._release_summary(
        campaign["release_dir"],
        {"release_label": "candidate"},
        {"north": {"required_tests": ["compare"]}},
        [],
        False,
        "new",
        catalogue["tests"],
    )

    assert summary[_PRODUCTION_HISTORY][0]["run_id"] == "run-1"


def test_write_summary_includes_failures_and_production_history(tmp_test_directory):
    report_dir = Path(tmp_test_directory)
    summary = {
        _RELEASE: "candidate",
        _QUALIFIED: False,
        _RESULTS: [
            {
                "site": "north",
                "test": "compare",
                _EXECUTION: "failed",
                _STATUS: "incomplete",
                "error": "missing report",
            }
        ],
        _PRODUCTION_HISTORY: [
            {
                "id": "production.north",
                _EXECUTION: "submitted",
                _STATUS: "submitted",
                "run_id": "run-1",
            }
        ],
    }

    science_test_reports._write_summary(report_dir, summary)

    markdown = (report_dir / "release-summary.md").read_text(encoding="utf-8")
    assert "incomplete | missing report" in markdown
    assert "Production history" in markdown
    assert "run-1" in markdown


def test_publication_filters_evidence_replaces_stale_output_and_warns_when_large(
    tmp_test_directory, monkeypatch, caplog
):
    root = Path(tmp_test_directory)
    work = root / "work"
    (work / "pair").mkdir(parents=True)
    (work / "pair/metrics.json").write_text("{}", encoding="utf-8")
    provenance = work / "provenance"
    provenance.mkdir()
    (provenance / "effective-settings.json").write_text("{}", encoding="utf-8")
    (provenance / "resolved-workflow.yml").write_text("workflow: {}\n", encoding="utf-8")
    (work / "inventory.json").write_text("[]", encoding="utf-8")
    (work / "comparison_inventory.json").write_text('{"matched_pair_count": 1}', encoding="utf-8")
    release_dir = root / _RELEASE
    report_root = release_dir / "reports/north/compare"
    report_root.mkdir(parents=True)
    (report_root / "stale.json").touch()
    result = {_STATUS: "pass"}
    monkeypatch.setattr(science_test_reports, "_PUBLICATION_WARNING_FILES", 0)

    with caplog.at_level(logging.WARNING):
        science_test_reports._write_test_report(work, report_root, release_dir, result)

    assert (report_root / "pair/metrics.json").read_text() == "{}"
    assert not (report_root / "stale.json").exists()
    assert result["publication"]["warning"]
    assert "Science-test publication contains" in caplog.text
    assert not (report_root / "provenance/effective-settings.json").exists()
    assert (report_root / "provenance/resolved-workflow.yml").is_file()
    assert not (report_root / "inventory.json").exists()
    assert (report_root / "comparison_inventory.json").is_file()


def test_runtime_change_retains_native_collection_and_invalidates_analysis():
    records = {
        "collect.north": {"site": "north", "test": "collect", "runtime": "host", _STATUS: "pass"},
        "compare.north": {
            "site": "north",
            "test": "compare",
            "runtime": "container",
            _STATUS: "warn",
        },
    }
    catalogue = {"collect": {"collect_production": True}, "compare": {}, "derive": {}}
    science_test_reports._invalidate_runtime_changes(
        records, [{"site": "north", "test": "derive", "runtime": "host"}], catalogue
    )
    assert records["collect.north"][_STATUS] == "pass"
    assert records["compare.north"][_STATUS] == "not_run"


def test_derived_evidence_change_preserves_collection(tmp_test_directory):
    root = Path(tmp_test_directory)
    derived = root / "north/derived/histograms.hdf5"
    derived.parent.mkdir(parents=True)
    derived.write_bytes(b"old histogram")
    context = {"__SCIENCE_CANDIDATE_PATH__": str(root)}
    previous = {"north": science_test_reports._evidence_fingerprints(context, "north")}
    derived.write_bytes(b"new histogram")
    current = {"north": science_test_reports._evidence_fingerprints(context, "north")}
    records = {
        "collect.north": {"site": "north", "test": "collect", _STATUS: "pass"},
        "compare.north": {"site": "north", "test": "compare", _STATUS: "warn"},
    }
    science_test_reports._invalidate_evidence_changes(
        records, previous, current, {"collect": {"collect_production": True}, "compare": {}}
    )
    assert records["collect.north"][_STATUS] == "pass"
    assert records["compare.north"][_STATUS] == "not_run"


def test_write_summary_links_report_paths(tmp_test_directory):
    report_dir = Path(tmp_test_directory)
    summary = {
        _RELEASE: "candidate",
        _QUALIFIED: False,
        _PRODUCTION_HISTORY: [],
        _RESULTS: [
            {
                "site": "north",
                "test": "compare",
                _EXECUTION: "completed",
                _STATUS: "warn",
                "report": "reports/north/compare",
            }
        ],
    }
    science_test_reports._write_summary(report_dir, summary)
    markdown = (report_dir / "release-summary.md").read_text(encoding="utf-8")
    assert "[report](north/compare)" in markdown


def test_dependency_decision_is_visible_in_saved_report(tmp_test_directory):
    from simtools.model_repository.compatibility import LEGACY_POLICY

    state = "planned"
    proof = {
        "target_simtools": "0.37.0",
        "version": "0.17.1",
        "source": "selected-models",
        "allowed": "==0.17.1",
        "policy": LEGACY_POLICY,
        _STATUS: "compatible",
        "initialization": "not checked",
    }
    summary = {
        _RELEASE: "compatibility-example",
        _QUALIFIED: False,
        _PRODUCTION_HISTORY: [],
        _RESULTS: [
            {
                "id": "compatibility.south",
                "site": "south",
                "test": "compatibility",
                _EXECUTION: state,
                _STATUS: state,
                _DEPENDENCY_COMPATIBILITY: proof,
            }
        ],
    }
    science_test_reports._write_summary(Path(tmp_test_directory), summary)
    markdown = next(Path(tmp_test_directory).glob("*.md")).read_text()
    assert "models 0.17.1 at `selected-models`" in markdown
    assert LEGACY_POLICY in markdown
    assert "initialization: not checked" in markdown


def test_development_production_cannot_be_qualified_by_passing_collection():
    from simtools.model_repository.compatibility import DEVELOPMENT_STATUS

    assert not science_test_reports._dependencies_qualified(
        [{_DEPENDENCY_COMPATIBILITY: {_STATUS: DEVELOPMENT_STATUS}}]
    )
    assert science_test_reports._dependencies_qualified([{}])

import json
import logging
from pathlib import Path

import yaml

from simtools.science_tests import reports as science_test_reports


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
                "results": [],
                "production_history": [
                    {
                        "id": "production.north",
                        "test": "production",
                        "site": "north",
                        "execution": "submitted",
                        "status": "submitted",
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

    assert summary["production_history"][0]["run_id"] == "run-1"


def test_write_summary_includes_failures_and_production_history(tmp_test_directory):
    report_dir = Path(tmp_test_directory)
    summary = {
        "release": "candidate",
        "qualified": False,
        "results": [
            {
                "site": "north",
                "test": "compare",
                "execution": "failed",
                "status": "incomplete",
                "error": "missing report",
                "report": "reports/north/compare",
                "artifacts": ["science-comparison.json", "science-comparison.png"],
            }
        ],
        "production_history": [
            {
                "id": "production.north",
                "execution": "submitted",
                "status": "submitted",
                "run_id": "run-1",
            }
        ],
    }

    science_test_reports._write_summary(report_dir, summary)

    markdown = (report_dir / "release-summary.md").read_text(encoding="utf-8")
    assert "incomplete | missing report" in markdown
    assert "[science-comparison.json](north/compare/science-comparison.json)" in markdown
    assert "Production history" in markdown
    assert "run-1" in markdown


def test_publication_copies_nested_evidence_and_warns_when_large(
    tmp_test_directory, monkeypatch, caplog
):
    root = Path(tmp_test_directory)
    work = root / "work"
    (work / "pair").mkdir(parents=True)
    (work / "pair/metrics.json").write_text("{}", encoding="utf-8")
    release_dir = root / "release"
    report_root = release_dir / "reports/north/compare"
    report_root.mkdir(parents=True)
    (report_root / "stale.json").touch()
    result = {"status": "pass"}
    monkeypatch.setattr(science_test_reports, "_PUBLICATION_WARNING_FILES", 0)

    with caplog.at_level(logging.WARNING):
        science_test_reports._write_test_report(work, report_root, release_dir, result)

    assert (report_root / "pair/metrics.json").read_text() == "{}"
    assert not (report_root / "stale.json").exists()
    assert result["publication"]["warning"]
    assert "pair/metrics.json" in result["artifacts"]
    assert "Science-test publication contains" in caplog.text


def test_publication_omits_duplicate_provenance_and_checksum_inventory(tmp_test_directory):
    root = Path(tmp_test_directory)
    work = root / "work"
    provenance = work / "provenance"
    provenance.mkdir(parents=True)
    (provenance / "effective-settings.json").write_text("{}", encoding="utf-8")
    (provenance / "resolved-workflow.yml").write_text("workflow: {}\n", encoding="utf-8")
    (work / "inventory.json").write_text("[]", encoding="utf-8")
    (work / "comparison_inventory.json").write_text('{"matched_pair_count": 1}', encoding="utf-8")
    release_dir = root / "release"
    report_root = release_dir / "reports/north/compare"
    result = {"status": "pass"}

    science_test_reports._write_test_report(work, report_root, release_dir, result)

    assert not (report_root / "provenance/effective-settings.json").exists()
    assert (report_root / "provenance/resolved-workflow.yml").is_file()
    assert not (report_root / "inventory.json").exists()
    assert (report_root / "comparison_inventory.json").is_file()

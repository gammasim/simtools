import json

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

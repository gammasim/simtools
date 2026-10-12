from pathlib import Path

import pytest
import yaml

from simtools.science_tests import definitions as science_test_definitions
from simtools.science_tests import execution as science_test_execution


def _write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.fixture
def campaign(tmp_test_directory):
    root = Path(tmp_test_directory)
    release = root / "release"
    template = root / "science-test-template"
    _write_yaml(
        release / "release.yml",
        {
            "release_label": "candidate",
            "baseline": "baseline",
            "required_sites": ["north", "south"],
        },
    )
    for site in ("north", "south"):
        _write_yaml(
            release / "sites" / f"{site}.yml",
            {
                "site": site,
                "array_layout_name": f"CTAO-{site.title()}-Alpha",
                "required_tests": ["compare"],
            },
        )
    _write_yaml(
        template / "catalogue.yml",
        {
            "tests": {
                "derive": {
                    "workflow": "workflow.yml",
                    "tier": "release-light",
                    "sites": ["north", "south"],
                },
                "compare": {
                    "workflow": "workflow.yml",
                    "tier": "release-light",
                    "sites": ["north", "south"],
                    "depends_on": ["derive"],
                    "acceptance_rule": "comparison",
                },
            }
        },
    )
    _write_yaml(
        template / "workflow.yml",
        {
            "schema_version": "0.7.0",
            "schema_name": "application_workflow.metaschema",
            "applications": [
                {
                    "application": "simtools-compare-productions",
                    "configuration": {"output_path": "__SCIENCE_REPORT_PATH__"},
                }
            ],
            "collection": {
                "output_path": "__SCIENCE_COLLECTION_PATH__",
                "files": ["metrics.json"],
                "write_inventory": False,
                "preserve_relative_paths": True,
            },
        },
    )
    _write_yaml(
        template / "acceptance" / "expected-products.yml",
        {
            "tests": {
                "derive": ["__SCIENCE_REPORT_PATH__/metrics.json"],
                "compare": ["__SCIENCE_REPORT_PATH__/metrics.json"],
            }
        },
    )
    _write_yaml(
        template / "acceptance" / "thresholds.yml",
        {
            "rules": {
                "comparison": {
                    "mode": "advisory",
                    "metrics": [
                        {
                            "file": "__SCIENCE_REPORT_PATH__/metrics.json",
                            "key": "count",
                            "minimum": 2,
                        }
                    ],
                },
            }
        },
    )
    context = root / "context.yml"
    _write_yaml(
        context,
        {
            key: str(root / name)
            for key, name in zip(
                science_test_definitions._PATH_KEYS, ("candidate", "baseline", "reference")
            )
        },
    )
    return {"release_dir": release, "template_dir": template, "context_file": context}


@pytest.fixture
def successful_workflow(monkeypatch):
    calls = []

    def execute(args, replacements):
        from simtools.runners.simtools_runner import _copy_collection_files

        calls.append(replacements["__SCIENCE_SITE_KEY__"])
        output = Path(replacements["__SCIENCE_REPORT_PATH__"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text('{"count": 3}', encoding="utf-8")
        _copy_collection_files(
            [{"configuration": {"output_path": str(output)}}],
            {
                "output_path": replacements["__SCIENCE_COLLECTION_PATH__"],
                "files": ["metrics.json"],
                "write_inventory": False,
            },
        )

    monkeypatch.setattr(science_test_execution, "run_applications", execute)
    return calls

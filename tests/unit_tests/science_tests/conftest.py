from pathlib import Path

import pytest
import yaml

from simtools.science_tests import definitions as defs
from simtools.science_tests import execution

ACCEPTANCE = "acceptance"
APPLICATION = "application"
COMPARE = "compare"
COMPARE_YML = "compare.yml"
COMPARISON = "comparison"
CONFIGURATION = "configuration"
CONTEXT_FILE = "context_file"
DERIVE = "derive"
OUTPUT_PATH = "output_path"
RELEASE_DIR = "release_dir"
SIMTOOLS_COMPARE_PRODUCTIONS = "simtools-compare-productions"
SITES_KEY = "sites"
TESTS = "tests"
UTF_8 = "utf-8"
WORKFLOW = "workflow"

WORKFLOW_FILE = "workflow.yml"
METRICS_FILE = "metrics.json"
REPORT_METRICS = f"{defs.REPORT_PATH}/{METRICS_FILE}"
SITES = ["north", "south"]
LIGHT = "release-light"


def write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding=UTF_8)
    return path


def workflow(*applications, **extra):
    return {
        "schema_version": "0.7.0",
        "schema_name": "application_workflow.metaschema",
        "applications": list(applications),
        **extra,
    }


@pytest.fixture(autouse=True)
def skip_comparison_input_validation(monkeypatch):
    """Keep orchestration tests independent of production metadata trees."""
    monkeypatch.setattr(execution, "_validate_comparison_inputs", lambda *_args: None)


@pytest.fixture
def campaign(tmp_test_directory):
    """Release with a 'derive' test and a 'compare' test reading baseline products."""
    root = Path(tmp_test_directory) / "campaign"
    release = root / "v2" / "science_tests"
    template = root / defs.TEMPLATE_DIRECTORY
    write_yaml(
        release / defs.RELEASE_FILE,
        {"release_label": "v2", "baseline": "v1", "required_sites": SITES, "model_version": "7"},
    )
    for site in SITES:
        write_yaml(
            template / SITES_KEY / f"{site}.yml",
            {"site": site, "array_layout_name": f"CTAO-{site}", "required_tests": [COMPARE]},
        )
    write_yaml(
        template / "catalogue.yml",
        {
            TESTS: {
                DERIVE: {WORKFLOW: WORKFLOW_FILE, "tier": LIGHT, SITES_KEY: SITES},
                COMPARE: {
                    WORKFLOW: COMPARE_YML,
                    "tier": LIGHT,
                    SITES_KEY: SITES,
                    "depends_on": [DERIVE],
                    "acceptance_rule": COMPARISON,
                },
            }
        },
    )
    application = {
        APPLICATION: SIMTOOLS_COMPARE_PRODUCTIONS,
        CONFIGURATION: {OUTPUT_PATH: defs.REPORT_PATH},
    }
    write_yaml(template / WORKFLOW_FILE, workflow(application))
    baseline_application = {
        APPLICATION: SIMTOOLS_COMPARE_PRODUCTIONS,
        CONFIGURATION: {
            OUTPUT_PATH: defs.REPORT_PATH,
            "baseline_path": defs.BASELINE_SITE_PATH,
        },
    }
    write_yaml(template / COMPARE_YML, workflow(baseline_application))
    write_yaml(
        template / ACCEPTANCE / "expected-products.yml",
        {TESTS: {DERIVE: [REPORT_METRICS], COMPARE: [REPORT_METRICS]}},
    )
    write_yaml(
        template / ACCEPTANCE / "thresholds.yml",
        {
            "rules": {
                COMPARISON: {
                    "mode": "advisory",
                    "metrics": [{"file": REPORT_METRICS, "key": "count", "minimum": 2}],
                }
            }
        },
    )
    context = write_yaml(
        release / defs.CONTEXT_FILE,
        {
            defs.CANDIDATE_PATH: str(root / "data" / "v2"),
            defs.BASELINE_PATH: str(root / "data" / "v1"),
            defs.PRODUCTION_CONFIGURATION_PATH: str(root / "data" / "production"),
        },
    )
    return {RELEASE_DIR: release, "template_dir": template, CONTEXT_FILE: context}


@pytest.fixture
def bundle(campaign):
    return defs.load_release(campaign[RELEASE_DIR], context_file=campaign[CONTEXT_FILE])


@pytest.fixture
def successful_workflow(monkeypatch):
    """Replace workflow execution by writing the expected metrics product."""
    calls = []

    def run(args, replacements):
        calls.append((replacements["__SCIENCE_SITE_KEY__"], args["config_file"]))
        output = Path(replacements[defs.REPORT_PATH])
        output.mkdir(parents=True, exist_ok=True)
        (output / METRICS_FILE).write_text('{"count": 3}', encoding=UTF_8)

    monkeypatch.setattr(execution, "run_applications", run)
    return calls

import json
import logging
from pathlib import Path

import pytest
import yaml

from simtools.science_tests import campaign as campaign_runner
from simtools.science_tests import definitions as defs

MODELS = "/models"
BASELINE_OLD = "baseline: old\n"
COMPARE_NORTH = "compare.north"
CONTEXT_FILE = "context_file"
RELEASE_DIR = "release_dir"
REPORTS = "reports"
RUN_ID = "run_id"
SCIENCE_TESTS = "science_tests"
SETUP = "setup"
SITE_NORTH = "site: north\n"
SITES = "sites"
SITES_NORTH_YML = "sites/north.yml"
SOUTH = "south"

ENCODING = "utf-8"
TEMPLATE = defs.TEMPLATE_DIRECTORY
CONTEXT_TEMPLATE = "context.example.yml"
NORTH = "north"
DERIVE = "derive"
COMPARE = "compare"
STATUS = "status"
RESULTS = "results"
BASELINE_VERSION = "v0.38.0"
MODEL_REVISION = "v0.18.0"


def write_files(root, files):
    for relative, contents in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding=ENCODING)


def results_by_id(summary):
    return {result["id"]: result for result in summary[RESULTS]}


def test_setup_release_creates_release_from_templates(tmp_test_directory, caplog):
    root = Path(tmp_test_directory) / SETUP
    release = root / "v1.2.3" / SCIENCE_TESTS
    write_files(
        root / TEMPLATE,
        {
            defs.RELEASE_FILE: f"release_label: {defs.RELEASE_LABEL}\n",
            "sites/east.yml": "site: east\n",
            "sites/west.yml": "---\nsite: west\n",
            CONTEXT_TEMPLATE: f"{defs.CANDIDATE_PATH}: /data/{defs.RELEASE_LABEL}\n",
        },
    )
    with caplog.at_level(logging.INFO):
        campaign_runner.setup_release(release)

    assert (release / defs.RELEASE_FILE).read_text(encoding=ENCODING) == (
        "---\nrelease_label: v1.2.3\n"
    )
    assert "/data/v1.2.3" in (release / defs.CONTEXT_FILE).read_text(encoding=ENCODING)
    assert sorted(path.name for path in (release / SITES).iterdir()) == ["east.yml", "west.yml"]
    assert "Created science-test setup" in caplog.text
    with pytest.raises(FileExistsError, match="already exist"):
        campaign_runner.setup_release(release)


def test_setup_release_overrides_baseline_and_models(tmp_test_directory):
    root = Path(tmp_test_directory) / SETUP
    release = root / "not a label" / SCIENCE_TESTS
    write_files(
        root / TEMPLATE,
        {
            defs.RELEASE_FILE: BASELINE_OLD,
            SITES_NORTH_YML: SITE_NORTH,
            CONTEXT_TEMPLATE: f"# Keep\n{defs.BASELINE_PATH}: ${{ROOT}}/old\n",
        },
    )
    campaign_runner.setup_release(
        release,
        baseline_version=BASELINE_VERSION,
        simulation_models_git_revision=MODEL_REVISION,
    )

    settings = yaml.safe_load((release / defs.RELEASE_FILE).read_text(encoding=ENCODING))
    assert settings == {"baseline": BASELINE_VERSION, defs.MODELS_GIT_REVISION: MODEL_REVISION}
    context = (release / defs.CONTEXT_FILE).read_text(encoding=ENCODING)
    assert "# Keep" in context
    assert yaml.safe_load(context)[defs.BASELINE_PATH] == f"${{ROOT}}/{BASELINE_VERSION}"


def test_setup_release_requires_templates_and_valid_baseline(tmp_test_directory):
    root = Path(tmp_test_directory) / SETUP
    template = root / TEMPLATE
    (template / SITES).mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="site templates not found"):
        campaign_runner.setup_release(root / "v1" / SCIENCE_TESTS)
    write_files(template, {SITES_NORTH_YML: SITE_NORTH, CONTEXT_TEMPLATE: "{}\n"})
    with pytest.raises(FileNotFoundError, match="setup templates not found"):
        campaign_runner.setup_release(root / "v1" / SCIENCE_TESTS)
    write_files(template, {defs.RELEASE_FILE: BASELINE_OLD})
    with pytest.raises(ValueError, match="requires __SCIENCE_BASELINE_PATH__"):
        campaign_runner.setup_release(root / "v1" / SCIENCE_TESTS, baseline_version="v0")
    for baseline in ("", "../baseline"):
        with pytest.raises(ValueError, match="filename-safe"):
            campaign_runner.setup_release(root, baseline_version=baseline)


def test_dry_run_plans_required_tests_without_writes(campaign, successful_workflow):
    summary = campaign_runner.run_release(**campaign, dry_run=True)

    assert {result[STATUS] for result in summary[RESULTS]} == {"planned"}
    assert len(summary[RESULTS]) == 2
    assert not successful_workflow
    assert not (campaign[RELEASE_DIR] / REPORTS).exists()


def test_run_writes_summary_and_keeps_subset_results(campaign, successful_workflow):
    first = campaign_runner.run_release(**campaign, tests=[DERIVE, COMPARE])
    second = campaign_runner.run_release(**campaign, sites=["SOUTH"], tests=[COMPARE])

    assert results_by_id(first)[COMPARE_NORTH][STATUS] == "warn"
    assert results_by_id(second)[COMPARE_NORTH][STATUS] == "warn"
    assert results_by_id(second)["compare.south"][RUN_ID] != first[RESULTS][0][RUN_ID]
    assert first["qualified"] is False
    reports = campaign[RELEASE_DIR] / REPORTS
    assert json.loads((reports / "release-summary.json").read_text(encoding=ENCODING))
    assert "acceptance.md" in {path.name for path in (reports / NORTH / COMPARE).iterdir()}


def test_rerunning_dependency_invalidates_dependent_result(campaign, successful_workflow):
    campaign_runner.run_release(**campaign, sites=[NORTH], tests=[DERIVE, COMPARE])
    summary = campaign_runner.run_release(**campaign, sites=[NORTH], tests=[DERIVE])
    assert results_by_id(summary)[COMPARE_NORTH][STATUS] == "not_run"


def test_run_without_baseline_tests_accepts_missing_baseline(campaign, successful_workflow):
    context = yaml.safe_load(campaign[CONTEXT_FILE].read_text(encoding=ENCODING))
    context[defs.BASELINE_PATH] = context[defs.CANDIDATE_PATH]
    campaign[CONTEXT_FILE].write_text(yaml.safe_dump(context), encoding=ENCODING)

    campaign_runner.run_release(**campaign, tests=[DERIVE])
    with pytest.raises(ValueError, match="non-overlapping"):
        campaign_runner.run_release(**campaign, tests=[COMPARE])


def test_run_raises_for_failed_tests(campaign, mocker):
    mocker.patch(
        "simtools.science_tests.execution.run_applications", side_effect=RuntimeError("boom")
    )
    with pytest.raises(RuntimeError, match=r"derive\.north: boom"):
        campaign_runner.run_release(**campaign, sites=[NORTH], tests=[DERIVE])


def test_production_requires_explicit_selection_and_preflight(campaign, mocker, caplog):
    catalogue_path = campaign["template_dir"] / "catalogue.yml"
    catalogue = yaml.safe_load(catalogue_path.read_text(encoding=ENCODING))
    catalogue["tests"][DERIVE]["produces_production"] = True
    catalogue_path.write_text(yaml.safe_dump(catalogue), encoding=ENCODING)
    with pytest.raises(ValueError, match="--allow_production"):
        campaign_runner.run_release(**campaign, tests=[DERIVE])

    mocker.patch("simtools.science_tests.execution._preview", return_value={})
    execute = mocker.patch.object(campaign_runner, "execute_tests", return_value=[])
    validate = mocker.patch.object(campaign_runner, "validate_context_paths")
    canary = mocker.patch.object(campaign_runner, "run_worker_canary")
    with caplog.at_level(logging.WARNING):
        campaign_runner.run_release(**campaign, tests=[DERIVE], allow_production=True)
    assert "without --preflight" in caplog.text
    campaign_runner.run_release(**campaign, tests=[DERIVE], allow_production=True, preflight=True)

    validate.assert_called_once()
    assert [call.args[0].site for call in canary.call_args_list] == [NORTH, SOUTH]
    assert canary.call_args.args[2].parts[-2:] == ("preflight", SOUTH)
    assert execute.call_count == 2


def test_run_applies_release_model_source(campaign, successful_workflow, mocker):
    release_path = campaign[RELEASE_DIR] / defs.RELEASE_FILE
    release = yaml.safe_load(release_path.read_text(encoding=ENCODING))
    release[defs.MODELS_PATH] = MODELS
    release_path.write_text(yaml.safe_dump(release), encoding=ENCODING)
    signature = mocker.spy(defs, "campaign_signature")

    campaign_runner.run_release(
        **campaign, sites=[NORTH], tests=[DERIVE], application_args={defs.MODELS_GIT_PATH: "/git"}
    )

    assert signature.call_args.args[1] == {defs.MODELS_PATH: MODELS}

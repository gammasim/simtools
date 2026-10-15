import json
import logging
import shutil
from pathlib import Path

import pytest
import yaml

from simtools.science_tests import campaign as campaign_runner
from simtools.science_tests import definitions as science_test_definitions
from simtools.science_tests import execution as science_test_execution

NOT_RUN_STATUS = "not_run"

BASELINE_KEY = "baseline"
MODEL_REVISION = "v0.18.0"
BASELINE_VERSION = "v0.38.0"
TEMPLATE_DIRECTORY = "science-test-template"
RELEASE_FILE = "release.yml"
CONTEXT_FILE = "context.yml"
CONTEXT_TEMPLATE = "context.example.yml"
BASELINE_PATH_KEY = "__SCIENCE_BASELINE_PATH__"
MODEL_REVISION_KEY = "simulation_models_git_revision"
PREVIOUS_RELEASE = "previous"
MODEL_REF = "models"
NORTH_SITE_FILE = "sites/north.yml"
TEXT_ENCODING = "utf-8"


def _write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding=TEXT_ENCODING)


def test_setup_release_creates_minimal_release_from_templates(tmp_test_directory, caplog):
    root = Path(tmp_test_directory)
    template = root / TEMPLATE_DIRECTORY
    release = root / "v1.2.3" / "science_tests"
    for relative, contents in {
        RELEASE_FILE: "release_label: __SCIENCE_RELEASE_LABEL__\n",
        "sites/east.yml": "site: east\n",
        "sites/west.yml": "site: west\n",
        CONTEXT_TEMPLATE: "__SCIENCE_CANDIDATE_PATH__: /path/candidate\n",
    }.items():
        path = template / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding=TEXT_ENCODING)

    with caplog.at_level(logging.INFO):
        campaign_runner.setup_release(release, template)

    assert (release / RELEASE_FILE).read_text(
        encoding=TEXT_ENCODING
    ) == "---\nrelease_label: v1.2.3\n"
    assert all(
        path.read_text(encoding=TEXT_ENCODING).startswith("---") for path in release.rglob("*.yml")
    )
    assert "Edit these files before the dry run" in caplog.text
    assert sorted(path.relative_to(release).as_posix() for path in release.rglob("*")) == [
        CONTEXT_FILE,
        RELEASE_FILE,
        "sites",
        "sites/east.yml",
        "sites/west.yml",
    ]


def test_setup_release_does_not_overwrite_existing_files(tmp_test_directory):
    root = Path(tmp_test_directory)
    template = root / TEMPLATE_DIRECTORY
    release = root / "v1.2.3" / "science_tests"
    (template / RELEASE_FILE).parent.mkdir(parents=True)
    (template / RELEASE_FILE).write_text("release_label: __SCIENCE_RELEASE_LABEL__\n")
    (template / "sites").mkdir()
    (template / "sites" / "example.yml").write_text("site: example\n")
    (template / CONTEXT_TEMPLATE).write_text("context\n")
    (release / RELEASE_FILE).parent.mkdir(parents=True)
    (release / RELEASE_FILE).write_text("existing\n")

    with pytest.raises(FileExistsError, match="already exist"):
        campaign_runner.setup_release(release, template)


def test_setup_release_requires_site_templates(tmp_test_directory):
    template = Path(tmp_test_directory) / TEMPLATE_DIRECTORY
    (template / "sites").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="site templates not found"):
        campaign_runner.setup_release(Path(tmp_test_directory) / "release/science_tests", template)


def test_setup_release_overrides_baseline_and_models(tmp_test_directory):
    root = Path(tmp_test_directory)
    template = root / TEMPLATE_DIRECTORY
    release = root / "v0.39.0/science_tests"
    for relative, contents in {
        RELEASE_FILE: "baseline: old\nsimulation_models_git_revision: old\n",
        NORTH_SITE_FILE: "site: north\n",
        CONTEXT_TEMPLATE: (
            "# Preserve this comment\n"
            "__SCIENCE_BASELINE_PATH__: ${SIMTOOLS_SCIENCE_TESTS_ROOT}/old\n"
        ),
    }.items():
        path = template / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding=TEXT_ENCODING)
    campaign_runner.setup_release(
        release,
        template,
        baseline_version=BASELINE_VERSION,
        simulation_models_git_revision=MODEL_REVISION,
    )
    settings = yaml.safe_load((release / RELEASE_FILE).read_text(encoding=TEXT_ENCODING))
    assert settings == {BASELINE_KEY: BASELINE_VERSION, MODEL_REVISION_KEY: MODEL_REVISION}
    context = (release / CONTEXT_FILE).read_text(encoding=TEXT_ENCODING)
    assert "# Preserve this comment" in context
    assert yaml.safe_load(context)[BASELINE_PATH_KEY] == ("${SIMTOOLS_SCIENCE_TESTS_ROOT}/v0.38.0")


@pytest.mark.parametrize(BASELINE_KEY, ["", "../baseline", "/absolute"])
def test_setup_release_rejects_invalid_baseline(tmp_test_directory, baseline):
    with pytest.raises(ValueError, match="filename-safe release label"):
        campaign_runner.setup_release(tmp_test_directory, baseline_version=baseline)


def test_setup_overrides_adds_missing_release_fields():
    contents = campaign_runner._setup_overrides("---\n", RELEASE_FILE, PREVIOUS_RELEASE, MODEL_REF)
    assert yaml.safe_load(contents) == {
        BASELINE_KEY: PREVIOUS_RELEASE,
        MODEL_REVISION_KEY: MODEL_REF,
    }


def test_setup_overrides_requires_baseline_path():
    with pytest.raises(ValueError, match="Setup context requires"):
        campaign_runner._setup_overrides("{}", CONTEXT_FILE, PREVIOUS_RELEASE, None)


def test_run_collects_evidence_and_preserves_subset_results(campaign, successful_workflow):
    planned = campaign_runner.run_release(**campaign, dry_run=True)
    assert all(result["status"] == "planned" for result in planned["results"])
    assert not planned["qualified"]
    assert successful_workflow == []
    assert not (campaign["release_dir"] / "reports").exists()

    north = campaign_runner.run_release(**campaign, sites=["north"], tests=["compare"])
    assert {r["id"]: r["status"] for r in north["results"]} == {
        "compare.north": "warn",
        "compare.south": NOT_RUN_STATUS,
    }
    south = campaign_runner.run_release(**campaign, sites=["south"], tests=["compare"])
    assert len(south["results"]) == 2
    assert successful_workflow == ["north", "south"]
    result = south["results"][0]
    report = campaign["release_dir"] / result["report"]
    assert json.loads((report / "metrics.json").read_text()) == {"count": 3}
    assert not (report / "inventory.json").exists()
    assert not (report / "execution.log").exists()
    assert not (report / "workflow.yml").exists()
    assert not south["qualified"]
    assert (
        campaign_runner.run_release(**campaign, dry_run=True)["results"][0]["status"] == "planned"
    )
    assert (
        json.loads((campaign["release_dir"] / "reports/release-summary.json").read_text()) == south
    )
    assert not (campaign["release_dir"] / "reports/north/metrics.json").exists()


@pytest.mark.parametrize(
    "selection", [{"tests": ["missing"]}, {"sites": ["missing"]}, {"tests": []}, {"sites": []}]
)
def test_invalid_selection_fails_without_execution(campaign, successful_workflow, selection):
    with pytest.raises(ValueError, match="selection"):
        campaign_runner.run_release(**campaign, **selection)
    assert successful_workflow == []


def test_missing_unselected_site_fails_preflight(campaign, successful_workflow):
    (campaign["release_dir"] / "sites/south.yml").unlink()
    with pytest.raises(FileNotFoundError):
        campaign_runner.run_release(**campaign, sites=["north"])
    assert successful_workflow == []


def test_campaign_signature_keeps_native_steps_across_runtime_choices(
    campaign, successful_workflow
):
    standard = campaign_runner._prepare_release_run(
        campaign["release_dir"],
        campaign["template_dir"],
        campaign["context_file"],
        None,
        None,
        False,
        {},
    )
    host = campaign_runner._prepare_release_run(
        campaign["release_dir"],
        campaign["template_dir"],
        campaign["context_file"],
        None,
        None,
        False,
        {"ignore_runtime_environment": True},
    )

    assert standard["signature"] == host["signature"]
    assert successful_workflow == []


def test_non_comparison_selection_does_not_require_a_baseline(campaign, successful_workflow):
    context = campaign["context_file"]
    configuration = yaml.safe_load(context.read_text(encoding=TEXT_ENCODING))
    del configuration[BASELINE_PATH_KEY]
    _write_yaml(context, configuration)

    summary = campaign_runner.run_release(**campaign, sites=["north"], tests=["derive"])

    assert [
        result["id"] for result in summary["results"] if result["status"] != NOT_RUN_STATUS
    ] == ["derive.north"]


@pytest.mark.parametrize(
    "configuration_file",
    [
        "workflow.yml",
        "run_time.yml",
        "candidate/events.hdf5",
        "baseline/events.hdf5",
        "candidate/metadata.yml",
    ],
)
def test_configuration_changes_invalidate_results_but_external_file_changes_do_not(
    campaign, successful_workflow, configuration_file
):
    role, _, name = configuration_file.partition("/")
    if name:
        context = science_test_definitions._load_yaml(campaign["context_file"])
        path = Path(context[f"__SCIENCE_{role.upper()}_PATH__"]) / "north" / name
        path.parent.mkdir(parents=True)
        path.write_text("old evidence\n", encoding=TEXT_ENCODING)
    else:
        path = campaign["template_dir"] / configuration_file
    campaign_runner.run_release(**campaign, sites=["north"])
    if configuration_file == "run_time.yml":
        _write_yaml(path, {"runtime_environment": {"image": "test"}})
    else:
        path.write_text(path.read_text() + "# changed workflow\n", encoding=TEXT_ENCODING)
    summary = campaign_runner.run_release(**campaign, sites=["south"])
    expected_status = "warn" if name else NOT_RUN_STATUS
    assert next(r for r in summary["results"] if r["id"] == "compare.north")["status"] == (
        expected_status
    )


def test_comparison_execution_uses_release_parameters(campaign, successful_workflow):
    release_path = campaign["release_dir"] / RELEASE_FILE
    release = yaml.safe_load(release_path.read_text(encoding=TEXT_ENCODING))
    release.update(
        expected_changes=["configuration.azimuth_angle"],
        comparison_parameters={"compare": {"compare_by": ["configuration.azimuth_angle"]}},
    )
    _write_yaml(release_path, release)
    summary = campaign_runner.run_release(**campaign, sites=["north"], tests=["compare"])
    workflow = Path(summary["results"][0]["execution_provenance"]) / "workflow.yml"
    compare_workflow = yaml.safe_load(workflow.read_text())["applications"][0]["configuration"]
    assert compare_workflow["compare_by"] == ["configuration.azimuth_angle"]
    assert successful_workflow == ["north"]


def test_rerunning_dependency_invalidates_comparison(campaign, successful_workflow):
    campaign_runner.run_release(**campaign, sites=["north"])
    summary = campaign_runner.run_release(**campaign, sites=["north"], tests=["derive"])
    record = next(r for r in summary["results"] if r["id"] == "compare.north")
    assert record["status"] == NOT_RUN_STATUS
    assert "Upstream" in record["reason"]


def test_two_checkouts_have_portable_equivalent_plans(campaign, tmp_test_directory):
    root = Path(tmp_test_directory) / "second-checkout"
    shutil.copytree(campaign["release_dir"], root / "release")
    shutil.copytree(campaign["template_dir"], root / TEMPLATE_DIRECTORY)
    first = campaign_runner.run_release(**campaign, dry_run=True)
    second = campaign_runner.run_release(
        release_dir=root / "release", context_file=campaign["context_file"], dry_run=True
    )
    assert first["campaign_sha256"] == second["campaign_sha256"]
    assert [(r["id"], r["status"]) for r in first["results"]] == [
        (r["id"], r["status"]) for r in second["results"]
    ]
    assert str(root) not in json.dumps(second)


def test_unknown_acceptance_rule_fails_preflight(campaign, successful_workflow):
    _write_yaml(campaign["template_dir"] / "acceptance/thresholds.yml", {"rules": {}})
    with pytest.raises(ValueError, match="Missing acceptance rule"):
        campaign_runner.run_release(**campaign, dry_run=True)
    assert successful_workflow == []


def test_optional_catalogue_test_can_be_selected(campaign, successful_workflow):
    path = campaign["template_dir"] / "catalogue.yml"
    catalogue = yaml.safe_load(path.read_text())
    catalogue["tests"]["grid"] = {
        "workflow": "workflow.yml",
        "sites": ["north"],
        "tier": "release-light",
    }
    _write_yaml(path, catalogue)
    products = campaign["template_dir"] / "acceptance/expected-products.yml"
    contracts = yaml.safe_load(products.read_text())
    contracts["tests"]["grid"] = ["__SCIENCE_REPORT_PATH__/metrics.json"]
    _write_yaml(products, contracts)
    summary = campaign_runner.run_release(**campaign, sites=["north"], tests=["grid"], dry_run=True)
    assert next(r for r in summary["results"] if r["id"] == "grid.north")["status"] == "planned"
    with pytest.raises(ValueError, match="does not support"):
        campaign_runner.run_release(**campaign, sites=["south"], tests=["grid"], dry_run=True)
    assert successful_workflow == []


def test_site_selection_is_case_insensitive_and_deduplicated(campaign, successful_workflow):
    summary = campaign_runner.run_release(**campaign, sites=["North", "NORTH"])
    assert successful_workflow == ["north"]
    assert {
        result["site"] for result in summary["results"] if result["execution"] == "completed"
    } == {"north"}


def test_campaign_model_settings(campaign, monkeypatch):
    context = science_test_definitions._load_yaml(campaign["context_file"])
    context["simulation_models_git_path"] = str(campaign["template_dir"] / MODEL_REF)
    _write_yaml(campaign["context_file"], context)
    release_file = campaign["release_dir"] / RELEASE_FILE
    release = science_test_definitions._load_yaml(release_file)
    release[MODEL_REVISION_KEY] = MODEL_REVISION
    _write_yaml(release_file, release)
    captured = []
    prepare = science_test_execution.prepare_workflow

    def capture(args, replacements):
        captured.append((args["simulation_models_git_path"], args[MODEL_REVISION_KEY]))
        return prepare(args, replacements=replacements)

    monkeypatch.setattr(science_test_execution, "prepare_workflow", capture)
    campaign_runner.run_release(
        **campaign,
        sites=["north"],
        dry_run=True,
        application_args={MODEL_REVISION_KEY: "main"},
    )
    assert captured == [(context["simulation_models_git_path"], MODEL_REVISION)]


def test_git_model_source_overrides_inherited_filesystem_path(campaign):
    context = science_test_definitions._load_yaml(campaign["context_file"])
    context["simulation_models_git_path"] = str(campaign["template_dir"] / "models.git")
    selected = science_test_definitions._model_source_arguments(
        {"simulation_models_path": str(campaign["template_dir"] / MODEL_REF)},
        context,
        {MODEL_REVISION_KEY: MODEL_REVISION},
    )
    assert selected["simulation_models_path"] is None
    assert selected["simulation_models_git_path"] == context["simulation_models_git_path"]
    assert selected[MODEL_REVISION_KEY] == MODEL_REVISION


def test_filesystem_model_source_overrides_inherited_git_source(campaign):
    models = str(campaign["template_dir"] / MODEL_REF)
    selected = science_test_definitions._model_source_arguments(
        {
            "simulation_models_git_path": str(campaign["template_dir"] / "models.git"),
            MODEL_REVISION_KEY: "main",
        },
        {"simulation_models_path": models},
        {},
    )
    assert selected["simulation_models_path"] == models
    assert selected["simulation_models_git_path"] is None
    assert selected[MODEL_REVISION_KEY] is None


def test_run_preflight_checks_host_paths_and_one_canary_per_site(monkeypatch, tmp_test_directory):
    candidate = Path(tmp_test_directory)
    prepared = [
        (
            (
                None,
                None,
                None,
                "North",
                None,
                None,
                {"workflow": "north-a.yml", "tier": "release-prod"},
            ),
            None,
        ),
        (
            (
                None,
                None,
                None,
                "North",
                None,
                None,
                {"workflow": "north-b.yml", "tier": "release-prod"},
            ),
            None,
        ),
        (
            (
                None,
                None,
                None,
                "South",
                None,
                None,
                {"workflow": "south.yml", "tier": "release-prod"},
            ),
            None,
        ),
    ]
    run = {
        "context": {"__SCIENCE_CANDIDATE_PATH__": str(candidate)},
        "template_dir": candidate / "templates",
    }
    validated = []
    canaries = []
    monkeypatch.setattr(
        campaign_runner, "validate_context_paths", lambda context: validated.append(context)
    )
    monkeypatch.setattr(
        campaign_runner,
        "run_worker_canary",
        lambda *args: canaries.append(args),
    )

    campaign_runner._run_preflight(prepared, run, preflight=True, dry_run=False)

    assert validated == [run["context"]]
    assert [call[1] for call in canaries] == ["north-a.yml", "south.yml"]
    assert [call[3] for call in canaries] == [
        candidate / "work/science-tests/preflight/North",
        candidate / "work/science-tests/preflight/South",
    ]


@pytest.mark.parametrize(
    ("preflight", "dry_run"),
    [(False, False), (True, True)],
)
def test_run_preflight_skips_validation_when_disabled_or_dry_run(monkeypatch, preflight, dry_run):
    monkeypatch.setattr(
        campaign_runner,
        "validate_context_paths",
        lambda _context: pytest.fail("host paths should not be checked"),
    )
    monkeypatch.setattr(
        campaign_runner,
        "run_worker_canary",
        lambda *_args: pytest.fail("worker canary should not run"),
    )

    campaign_runner._run_preflight([], {"context": {}}, preflight, dry_run)


def test_shared_site_defaults_allow_missing_release_site_files(campaign, successful_workflow):
    site_file = campaign["release_dir"] / NORTH_SITE_FILE
    shared = campaign["template_dir"] / "sites"
    shared.mkdir()
    shutil.copyfile(site_file, shared / "north.yml")
    site_file.unlink()
    summary = campaign_runner.run_release(**campaign, sites=["north"], dry_run=True)
    assert summary["required_sites"] == ["north", "south"]
    assert successful_workflow == []

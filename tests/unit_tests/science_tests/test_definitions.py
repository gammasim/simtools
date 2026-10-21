from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from simtools.science_tests import definitions as defs

EXPLICIT = "/explicit"
IMAGE_SIF = "/image.sif"
MODELS = "/models"
WORK = "/work"
CUSTOM = "__CUSTOM__"
IMAGE = "__IMAGE__"
ABSOLUTE_PATH = "absolute path"
ANALYSIS = "analysis"
ARRAY_LAYOUT_NAME = "array_layout_name"
COMPARE_BY = "compare_by"
CONFIGURATION_OVERRIDES = "configuration_overrides"
DEPENDS_ON = "depends_on"
IMAGE_KEY = "image"
OVERRIDE = "override"
RELEASE_DIR = "release_dir"
REPLACEMENTS = "replacements"
REQUIRED_TESTS = "required_tests"
SOUTH = "south"
TEMPLATE_DIR = "template_dir"
UNSET = "unset"
USES_BASELINE = "uses_baseline"
UTF_8 = "utf-8"
WORKFLOW_YML = "workflow.yml"

CATALOGUE = "catalogue.yml"
NORTH = "north"
COMPARE = "compare"
DERIVE = "derive"
SITE_TEMPLATE = "sites/north.yml"


def write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding=UTF_8)
    return path


def update_yaml(path, **values):
    data = yaml.safe_load(path.read_text(encoding=UTF_8))
    data.update(values)
    write_yaml(path, data)


def test_load_release_merges_site_overrides(campaign):
    write_yaml(
        campaign[RELEASE_DIR] / SITE_TEMPLATE,
        {"site": NORTH, ARRAY_LAYOUT_NAME: OVERRIDE, REQUIRED_TESTS: [DERIVE]},
    )
    bundle = defs.load_release(campaign[RELEASE_DIR])

    assert bundle.template_dir == campaign[TEMPLATE_DIR].resolve()
    assert bundle.suites[NORTH][ARRAY_LAYOUT_NAME] == OVERRIDE
    assert bundle.suites[SOUTH][REQUIRED_TESTS] == [COMPARE]
    assert set(bundle.catalogue) == {DERIVE, COMPARE}


@pytest.mark.parametrize(
    ("site_definition", "message"),
    [
        ({"site": SOUTH, ARRAY_LAYOUT_NAME: "a", REQUIRED_TESTS: ["x"]}, "declares site"),
        ({"site": NORTH, ARRAY_LAYOUT_NAME: "a", REQUIRED_TESTS: ["x"]}, "Unknown"),
    ],
)
def test_load_release_rejects_invalid_sites(campaign, site_definition, message):
    write_yaml(campaign[TEMPLATE_DIR] / SITE_TEMPLATE, site_definition)
    with pytest.raises(ValueError, match=message):
        defs.load_release(campaign[RELEASE_DIR])


def test_load_release_rejects_unsupported_site_and_unknown_files(campaign):
    catalogue = campaign[TEMPLATE_DIR] / CATALOGUE
    data = yaml.safe_load(catalogue.read_text(encoding=UTF_8))
    data["tests"][COMPARE]["sites"] = [SOUTH]
    write_yaml(catalogue, data)
    with pytest.raises(ValueError, match="does not support site"):
        defs.load_release(campaign[RELEASE_DIR])
    (campaign[TEMPLATE_DIR] / SITE_TEMPLATE).unlink()
    with pytest.raises(FileNotFoundError, match="No site definition"):
        defs.load_release(campaign[RELEASE_DIR])
    with pytest.raises(FileNotFoundError, match="not found"):
        defs.load_release(campaign[RELEASE_DIR] / "missing", campaign[TEMPLATE_DIR])


def test_load_release_rejects_placeholder_release_label(campaign):
    update_yaml(campaign[RELEASE_DIR] / defs.RELEASE_FILE, release_label=defs.RELEASE_LABEL)
    with pytest.raises(ValueError, match="filename-safe"):
        defs.load_release(campaign[RELEASE_DIR])


def test_resolve_template_dir_requires_template(tmp_test_directory):
    with pytest.raises(FileNotFoundError, match="No science-test-template"):
        defs.resolve_template_dir(Path(tmp_test_directory) / "a" / "b")


def test_load_context_resolves_storage_root(tmp_test_directory, monkeypatch):
    context = write_yaml(
        Path(tmp_test_directory) / "context.yml",
        {defs.CANDIDATE_PATH: f"${{{defs.SCIENCE_TESTS_ROOT_ENV}}}/v1", "unused": None},
    )
    monkeypatch.setenv(defs.SCIENCE_TESTS_ROOT_ENV, "/data/")
    assert defs.load_context(context)[defs.CANDIDATE_PATH] == "/data/v1"
    monkeypatch.setenv(defs.SCIENCE_TESTS_ROOT_ENV, "relative")
    with pytest.raises(ValueError, match=ABSOLUTE_PATH):
        defs.load_context(context)
    write_yaml(context, {defs.CANDIDATE_PATH: EXPLICIT})
    assert defs.load_context(context) == {defs.CANDIDATE_PATH: EXPLICIT}


def test_select_matches_case_insensitively():
    assert defs.select(None, ["a", "b"], "site") == ["a", "b"]
    assert defs.select(["B", "b"], ["a", "b"], "site") == ["b"]
    with pytest.raises(ValueError, match="Invalid site selection"):
        defs.select(["c"], ["a", "b"], "site")
    with pytest.raises(ValueError, match="Invalid test selection"):
        defs.select([], ["a"], "test")


def test_dependency_order_detects_cycles_and_unknown_tests():
    catalogue = {"a": {DEPENDS_ON: ["b"]}, "b": {}, "c": {DEPENDS_ON: ["c"]}}
    assert defs.dependency_order(["a", "b"], catalogue) == ["b", "a"]
    with pytest.raises(ValueError, match="Circular"):
        defs.dependency_order(["c"], catalogue)
    with pytest.raises(ValueError, match="Unknown science test"):
        defs.dependency_order(["d"], catalogue)


def test_test_definition_adds_contracts_and_baseline_usage(bundle):
    bundle.release[CONFIGURATION_OVERRIDES] = {COMPARE: {COMPARE_BY: []}}
    compare = defs.test_definition(bundle, COMPARE)
    derive = defs.test_definition(bundle, DERIVE)

    assert compare[USES_BASELINE] is True
    assert derive[USES_BASELINE] is False
    assert compare["rule"]["mode"] == "advisory"
    assert compare[CONFIGURATION_OVERRIDES] == {COMPARE_BY: []}
    assert derive["workflow_path"] == bundle.template_dir / WORKFLOW_YML


def test_test_definition_requires_products_and_rules(bundle):
    del bundle.products[DERIVE]
    with pytest.raises(ValueError, match="expected-product contract"):
        defs.test_definition(bundle, DERIVE)
    bundle.rules.clear()
    with pytest.raises(ValueError, match="Missing acceptance rule"):
        defs.test_definition(bundle, COMPARE)


def test_validate_context_checks_paths(bundle):
    defs.validate_context(bundle, uses_baseline=True)
    bundle.context[defs.BASELINE_PATH] = bundle.context[defs.CANDIDATE_PATH] + "/sub"
    with pytest.raises(ValueError, match="non-overlapping"):
        defs.validate_context(bundle, uses_baseline=True)
    defs.validate_context(bundle, uses_baseline=False)
    bundle.context[defs.CANDIDATE_PATH] = f"/storage/{defs.RELEASE_LABEL}"
    with pytest.raises(ValueError, match=ABSOLUTE_PATH):
        defs.validate_context(bundle, uses_baseline=False)


def test_site_replacements_include_release_and_model_source(bundle):
    bundle.context[defs.MODELS_GIT_PATH] = MODELS
    bundle.context[UNSET] = None
    replacements = defs.site_replacements(bundle, NORTH, Path(WORK))

    assert replacements["__SCIENCE_SITE__"] == "North"
    assert replacements[defs.BASELINE_LABEL] == "v1"
    assert replacements[defs.RELEASE_LABEL] == "v2"
    assert replacements["__SCIENCE_MODEL_VERSION__"] == "7"
    assert replacements["__SCIENCE_MODELS_SOURCE_PATH__"] == MODELS
    assert replacements[defs.REPORT_PATH] == WORK
    assert replacements[defs.CANDIDATE_SITE_PATH].endswith("/v2/north")
    assert UNSET not in replacements
    bundle.suites[NORTH][REPLACEMENTS] = {CUSTOM: 1}
    assert defs.site_replacements(bundle, NORTH, Path(WORK))[CUSTOM] == "1"
    bundle.suites[NORTH][REPLACEMENTS] = {defs.REPORT_PATH: "x"}
    with pytest.raises(ValueError, match="reserved keys"):
        defs.site_replacements(bundle, NORTH, Path(WORK))


@pytest.mark.parametrize(
    ("release", "context", "expected"),
    [
        ({}, {defs.MODELS_GIT_PATH: "/git"}, {defs.MODELS_PATH: None}),
        (
            {defs.MODELS_PATH: MODELS},
            {defs.MODELS_GIT_PATH: "/git"},
            {defs.MODELS_GIT_PATH: None, defs.MODELS_GIT_REVISION: None},
        ),
        ({}, {}, {}),
    ],
)
def test_model_source_arguments_select_one_source(release, context, expected):
    bundle = SimpleNamespace(release=release, context=context)
    args = defs.model_source_arguments({defs.MODELS_GIT_REVISION: "main"}, bundle)
    for key, value in expected.items():
        assert args[key] == value


def test_shared_runtime_environment_selects_role(tmp_test_directory):
    template = Path(tmp_test_directory) / "runtime"
    template.mkdir()
    assert defs.shared_runtime_environment(template, {}) is None
    write_yaml(
        template / defs.RUNTIME_FILE,
        {
            "runtime_environment": {"container_engine": "apptainer", IMAGE_KEY: IMAGE},
            "runtime_environments": {ANALYSIS: {IMAGE_KEY: "__SCIENCE_UNSET__"}},
        },
    )
    runtime = defs.shared_runtime_environment(template, {IMAGE: IMAGE_SIF})
    assert runtime[IMAGE_KEY] == IMAGE_SIF
    with pytest.raises(ValueError, match="Unresolved placeholder"):
        defs.shared_runtime_environment(template, {}, ANALYSIS)
    with pytest.raises(ValueError, match="No runtime environment configured"):
        defs.shared_runtime_environment(template, {}, "camera")


def test_resolve_workflow_prefers_release_override(bundle):
    override = write_yaml(bundle.release_dir / WORKFLOW_YML, {"applications": []})
    assert defs.resolve_workflow(bundle, WORKFLOW_YML) == override
    assert defs.resolve_workflow(bundle, "compare.yml").parent == bundle.template_dir
    for invalid in ("/absolute.yml", "../outside.yml"):
        with pytest.raises(ValueError, match="relative template path"):
            defs.resolve_workflow(bundle, invalid)


def test_campaign_signature_tracks_workflows_and_profiles(bundle):
    signature = defs.campaign_signature(bundle, {})
    assert defs.campaign_signature(bundle, {}) == signature
    write_yaml(bundle.template_dir / "profiles" / "local.yml", {"backend": "local"})
    assert defs.campaign_signature(bundle, {}) != signature


def test_load_yaml_requires_mapping(tmp_test_directory):
    path = Path(tmp_test_directory) / "list.yml"
    path.write_text("- item\n", encoding=UTF_8)
    with pytest.raises(ValueError, match="Expected a mapping"):
        defs.load_yaml(path)

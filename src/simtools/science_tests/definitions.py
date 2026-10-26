"""Load and validate science-test release definitions.

A release directory contains ``release.yml``, ``context.yml`` and optional site overrides.
Test definitions, workflows, site defaults, and acceptance contracts are read from the shared
science-test template. Workflows reference release values through the placeholders defined
in this module.
"""

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from simtools.constants import RUN_TIME_ENVIRONMENT_SCHEMA, SCIENCE_TEST_SCHEMA_PATH
from simtools.data_model import schema
from simtools.io import ascii_handler
from simtools.utils.general import replace_placeholders_recursively

CANDIDATE_PATH = "__SCIENCE_CANDIDATE_PATH__"
BASELINE_PATH = "__SCIENCE_BASELINE_PATH__"
CANDIDATE_SITE_PATH = "__SCIENCE_CANDIDATE_SITE_PATH__"
BASELINE_SITE_PATH = "__SCIENCE_BASELINE_SITE_PATH__"
PRODUCTION_CONFIGURATION_PATH = "__PRODUCTION_CONFIGURATION_PATH__"
REPORT_PATH = "__SCIENCE_REPORT_PATH__"
RELEASE_LABEL = "__SCIENCE_RELEASE_LABEL__"
BASELINE_LABEL = "__SCIENCE_BASELINE_LABEL__"
TELESCOPE = "__SCIENCE_CAMERA_TELESCOPE__"
CONFIG_DIRECTORY = "__CONFIG_DIRECTORY__"
_BASELINE_PLACEHOLDER = "__SCIENCE_BASELINE"
_UNRESOLVED_PLACEHOLDER = "__SCIENCE_"

MODELS_PATH = "simulation_models_path"
MODELS_GIT_PATH = "simulation_models_git_path"
MODELS_GIT_REVISION = "simulation_models_git_revision"
MODEL_SOURCE_KEYS = (MODELS_PATH, MODELS_GIT_PATH, MODELS_GIT_REVISION)

RELEASE_FILE = "release.yml"
CONTEXT_FILE = "context.yml"
TEMPLATE_DIRECTORY = "science-test-template"
RUNTIME_FILE = "run_time.yml"
VALID_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
SCIENCE_TESTS_ROOT_ENV = "SIMTOOLS_SCIENCE_TESTS_ROOT"
_ENCODING = "utf-8"
_TESTS_KEY = "tests"
_SITES_KEY = "sites"
_RELEASE_LABEL_KEY = "release_label"
_REPLACEMENTS_KEY = "replacements"
_WORKFLOW_KEY = "workflow"
_REQUIRED_TESTS_KEY = "required_tests"

logger = logging.getLogger(__name__)


@dataclass
class ReleaseDefinition:
    """Validated definitions of one release, as returned by :func:`load_release`."""

    release_dir: Path
    template_dir: Path
    release: dict
    catalogue: dict
    context: dict
    products: dict
    rules: dict
    suites: dict = field(default_factory=dict)


def load_release(release_dir, template_dir=None, context_file=None):
    """Load and validate all definitions of one release.

    Parameters
    ----------
    release_dir : str or pathlib.Path
        Release science-test directory.
    template_dir : str or pathlib.Path or None
        Shared template; discovered above the release directory by default.
    context_file : str or pathlib.Path or None
        Context file; ``<release_dir>/context.yml`` by default.

    Returns
    -------
    ReleaseDefinition
        Release, catalogue, context, site suites, and acceptance contracts.
    """
    release_dir = Path(release_dir).resolve()
    template_dir = resolve_template_dir(release_dir, template_dir)
    release = load_yaml(release_dir / RELEASE_FILE)
    validate_config(release, "release.schema.yml")
    if not VALID_NAME.fullmatch(release[_RELEASE_LABEL_KEY]):
        raise ValueError(f"release_label must be a filename-safe label: {release_dir}")
    catalogue_path = Path(release.get("catalogue", template_dir / "catalogue.yml"))
    catalogue = load_yaml(release_dir / catalogue_path)
    validate_config(catalogue, "catalogue.schema.yml")
    acceptance = template_dir / "acceptance"
    products = load_yaml(acceptance / "expected-products.yml")
    validate_config(products, "expected_products.schema.yml")
    thresholds = load_yaml(acceptance / "thresholds.yml")
    validate_config(thresholds, "thresholds.schema.yml")
    bundle = ReleaseDefinition(
        release_dir=release_dir,
        template_dir=template_dir,
        release=release,
        catalogue=catalogue[_TESTS_KEY],
        context=load_context(context_file or release_dir / CONTEXT_FILE),
        products=products[_TESTS_KEY],
        rules=thresholds["rules"],
    )
    bundle.suites = {
        site: _load_suite(bundle, site) for site in dict.fromkeys(release["required_sites"])
    }
    return bundle


def resolve_template_dir(release_dir, template_dir=None):
    """Return the given template directory or the nearest one above the release.

    Parameters
    ----------
    release_dir : pathlib.Path
        Release directory whose ancestors are searched.
    template_dir : str or pathlib.Path or None
        Explicit template directory.

    Returns
    -------
    pathlib.Path
        Shared science-test template directory.
    """
    if template_dir is not None:
        return Path(template_dir).resolve()
    for parent in Path(release_dir).resolve().parents:
        if (parent / TEMPLATE_DIRECTORY).is_dir():
            return parent / TEMPLATE_DIRECTORY
    raise FileNotFoundError(f"No {TEMPLATE_DIRECTORY} directory found above {release_dir}.")


def load_context(context_file):
    """Load context paths, resolving the shared storage root environment variable.

    Parameters
    ----------
    context_file : str or pathlib.Path
        Context YAML file.

    Returns
    -------
    dict
        Context values with ``${SIMTOOLS_SCIENCE_TESTS_ROOT}`` resolved.
    """
    context = load_yaml(Path(context_file))
    validate_config(context, "context.schema.yml")
    placeholder = f"${{{SCIENCE_TESTS_ROOT_ENV}}}"
    if placeholder not in json.dumps(context):
        return context
    root = os.environ.get(SCIENCE_TESTS_ROOT_ENV)
    if not root or not Path(root).is_absolute():
        raise ValueError(
            f"Set {SCIENCE_TESTS_ROOT_ENV} to an absolute path before loading {context_file}."
        )
    return replace_placeholders_recursively(context, {placeholder: str(Path(root))})


def select(requested, available, kind):
    """Return the requested names (all available ones by default), matching case-insensitively.

    Parameters
    ----------
    requested : iterable of str or None
        Requested names.
    available : iterable of str
        Valid names.
    kind : str
        Name kind used in error messages.

    Returns
    -------
    list of str
        Unique selected names.
    """
    by_name = {name.casefold(): name for name in available}
    if requested is None:
        return list(by_name.values())
    selected = list(dict.fromkeys(by_name.get(name.casefold(), name) for name in requested))
    unknown = [name for name in selected if name not in by_name.values()]
    if not selected or unknown:
        raise ValueError(f"Invalid {kind} selection {unknown}; available: {sorted(available)}")
    return selected


def dependency_order(test_ids, catalogue):
    """Return the tests and all their dependencies once each, dependencies first.

    Parameters
    ----------
    test_ids : iterable of str
        Test names.
    catalogue : dict
        Test definitions by name.

    Returns
    -------
    list of str
        Ordered test names.
    """
    ordered = []
    visiting = set()

    def visit(test_id):
        if test_id in ordered:
            return
        if test_id in visiting:
            raise ValueError(f"Circular science-test dependency involving {test_id!r}.")
        if test_id not in catalogue:
            raise ValueError(f"Unknown science test: {test_id!r}.")
        visiting.add(test_id)
        for dependency in catalogue[test_id].get("depends_on", []):
            visit(dependency)
        visiting.remove(test_id)
        ordered.append(test_id)

    for test_id in test_ids:
        visit(test_id)
    return ordered


def test_definition(bundle, test_id):
    """Return one catalogue entry with its acceptance contracts and resolved workflow path.

    Parameters
    ----------
    bundle : ReleaseDefinition
        Release definitions.
    test_id : str
        Test name.

    Returns
    -------
    dict
        Catalogue entry extended by ``workflow_path``, ``products``, ``rule``,
        ``uses_baseline``, and release ``configuration_overrides``.
    """
    definition = dict(bundle.catalogue[test_id])
    if test_id not in bundle.products:
        raise ValueError(f"Missing expected-product contract for {test_id}.")
    rule_id = definition.get("acceptance_rule")
    if rule_id and rule_id not in bundle.rules:
        raise ValueError(f"Missing acceptance rule {rule_id} for {test_id}.")
    workflow = resolve_workflow(bundle, definition[_WORKFLOW_KEY])
    definition.update(
        workflow_path=workflow,
        products=bundle.products[test_id],
        rule=bundle.rules.get(rule_id, {}),
        uses_baseline=_BASELINE_PLACEHOLDER
        in workflow.read_text(encoding=_ENCODING) + json.dumps(definition),
        configuration_overrides=bundle.release.get("configuration_overrides", {}).get(test_id, {}),
    )
    return definition


def validate_context(bundle, uses_baseline):
    """Require absolute context paths and separate candidate and baseline directories.

    Parameters
    ----------
    bundle : ReleaseDefinition
        Release definitions.
    uses_baseline : bool
        Whether any selected test reads baseline products.
    """
    context = bundle.context
    required = (CANDIDATE_PATH, PRODUCTION_CONFIGURATION_PATH)
    for key in (*required, BASELINE_PATH) if uses_baseline else required:
        value = context.get(key)
        if (
            not isinstance(value, str)
            or not Path(value).is_absolute()
            or _UNRESOLVED_PLACEHOLDER in value
        ):
            raise ValueError(f"Context requires an absolute path for {key}.")
    if not uses_baseline:
        return
    candidate = Path(context[CANDIDATE_PATH]).resolve()
    baseline = Path(context[BASELINE_PATH]).resolve()
    if candidate == baseline or candidate in baseline.parents or baseline in candidate.parents:
        raise ValueError(
            "Candidate and baseline paths must be separate, non-overlapping directories: "
            f"{candidate}; {baseline}."
        )


def site_replacements(bundle, site, report_path):
    """Return placeholder values for one site.

    Parameters
    ----------
    bundle : ReleaseDefinition
        Release definitions.
    site : str
        Site key.
    report_path : pathlib.Path
        Work directory of the test.

    Returns
    -------
    dict
        Placeholder strings mapped to their values.
    """
    context, release, suite = bundle.context, bundle.release, bundle.suites[site]
    replacements = {key: str(value) for key, value in context.items() if value is not None}
    model_source = context.get(MODELS_PATH) or context.get(MODELS_GIT_PATH)
    if model_source:
        replacements["__SCIENCE_MODELS_SOURCE_PATH__"] = str(model_source)
    for key, placeholder in (
        ("model_version", "__SCIENCE_MODEL_VERSION__"),
        ("baseline", BASELINE_LABEL),
    ):
        if release.get(key):
            replacements[placeholder] = str(release[key])
    replacements.update(
        {
            RELEASE_LABEL: str(release[_RELEASE_LABEL_KEY]),
            "__SCIENCE_SITE_KEY__": site,
            "__SCIENCE_SITE__": suite.get("site_label", site.title()),
            "__SCIENCE_ARRAY_LAYOUT__": suite["array_layout_name"],
            CANDIDATE_SITE_PATH: str(Path(context.get(CANDIDATE_PATH, "")) / site),
            BASELINE_SITE_PATH: str(Path(context.get(BASELINE_PATH, "")) / site),
            REPORT_PATH: str(report_path),
        }
    )
    site_values = {key: str(value) for key, value in suite.get(_REPLACEMENTS_KEY, {}).items()}
    reserved = set(replacements) & set(site_values)
    if reserved:
        raise ValueError(f"Site replacements cannot override reserved keys: {sorted(reserved)}")
    return replacements | site_values


def model_source_arguments(application_args, bundle):
    """Apply context and release model-source settings over runner arguments.

    The first of release, context, and runner arguments defining a model path selects the
    source type; settings of the other source type are cleared.

    Parameters
    ----------
    application_args : dict or None
        Runner arguments.
    bundle : ReleaseDefinition
        Release definitions.

    Returns
    -------
    dict
        Updated copy of the runner arguments.
    """
    release, context = bundle.release, bundle.context
    args = dict(application_args or {})
    for configuration in (context, release):
        args.update({key: configuration[key] for key in MODEL_SOURCE_KEYS if key in configuration})
    source = next(
        (
            configuration
            for configuration in (release, context, args)
            if configuration.get(MODELS_GIT_PATH) or configuration.get(MODELS_PATH)
        ),
        {},
    )
    if source.get(MODELS_GIT_PATH):
        args[MODELS_PATH] = None
    elif source.get(MODELS_PATH):
        args[MODELS_GIT_PATH] = None
        args[MODELS_GIT_REVISION] = None
    return args


def shared_runtime_environment(template_dir, replacements, runtime_role=None):
    """Return the shared runtime environment, or the one selected by ``runtime_role``.

    Parameters
    ----------
    template_dir : pathlib.Path
        Template directory containing the optional ``run_time.yml``.
    replacements : dict
        Placeholder values.
    runtime_role : str or None
        Key in ``runtime_environments``; ``runtime_environment`` is used when omitted.

    Returns
    -------
    dict or None
        Runtime environment; None if no runtime file exists and no role is requested.
    """
    runtime_file = template_dir / RUNTIME_FILE
    if not runtime_file.is_file() and not runtime_role:
        return None
    configuration = replace_placeholders_recursively(
        load_yaml(runtime_file), replacements | {CONFIG_DIRECTORY: str(template_dir)}
    )
    schema.validate_dict_using_schema(
        configuration, schema_file=RUN_TIME_ENVIRONMENT_SCHEMA, offline=True
    )
    if runtime_role:
        runtime = configuration.get("runtime_environments", {}).get(runtime_role)
        if runtime is None:
            raise ValueError(f"No runtime environment configured for role {runtime_role!r}.")
    else:
        runtime = configuration.get("runtime_environment")
    if _UNRESOLVED_PLACEHOLDER in json.dumps(runtime):
        raise ValueError(f"Unresolved placeholder in runtime environment: {runtime}")
    return runtime


def resolve_workflow(bundle, workflow):
    """Return a workflow file, preferring a release override over the template.

    Parameters
    ----------
    bundle : ReleaseDefinition
        Release definitions.
    workflow : str
        Workflow path relative to the template directory.

    Returns
    -------
    pathlib.Path
        Workflow file.
    """
    path = Path(workflow)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Workflow must be a relative template path: {workflow}")
    override = bundle.release_dir / path
    return override if override.is_file() else bundle.template_dir / path


def campaign_signature(bundle, model_source):
    """Return a SHA-256 digest of all definitions and the external context of a campaign.

    Parameters
    ----------
    bundle : ReleaseDefinition
        Release definitions.
    model_source : dict
        Effective model-source settings.

    Returns
    -------
    str
        Hex digest.
    """
    definitions = asdict(bundle)
    payload = [value for key, value in definitions.items() if not key.endswith("_dir")]
    payload.append(model_source)
    for definition in bundle.catalogue.values():
        payload.append(resolve_workflow(bundle, definition[_WORKFLOW_KEY]).read_text(_ENCODING))
    template_dir = bundle.template_dir
    for path in [template_dir / RUNTIME_FILE, *sorted((template_dir / "profiles").glob("*"))]:
        if path.is_file():
            payload.append(path.read_text(encoding=_ENCODING))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode(_ENCODING)).hexdigest()


def validate_config(configuration, schema_file):
    """Validate one science-test configuration against its packaged schema.

    Parameters
    ----------
    configuration : dict
        Configuration to validate.
    schema_file : str
        Schema file name in the science-test schema directory.
    """
    schema.validate_dict_using_schema(
        configuration, schema_file=SCIENCE_TEST_SCHEMA_PATH / schema_file, offline=True
    )


def load_yaml(path):
    """Load a YAML mapping.

    Parameters
    ----------
    path : pathlib.Path
        YAML file.

    Returns
    -------
    dict
        File contents.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Science-test configuration not found: {path}")
    data = ascii_handler.collect_data_from_file(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def _load_suite(bundle, site):
    """Load site defaults from the template and apply the release override."""
    suite = {}
    for directory in (bundle.template_dir, bundle.release_dir):
        path = directory / _SITES_KEY / f"{site}.yml"
        if path.is_file():
            suite.update(load_yaml(path))
    if not suite:
        raise FileNotFoundError(f"No site definition found for {site}.")
    validate_config(suite, "site.schema.yml")
    if suite["site"] != site:
        raise ValueError(f"Site definition {site} declares site {suite['site']!r}.")
    for test_id in dependency_order(suite[_REQUIRED_TESTS_KEY], bundle.catalogue):
        if site not in bundle.catalogue[test_id][_SITES_KEY]:
            raise ValueError(f"Test {test_id!r} does not support site {site!r}.")
    return suite

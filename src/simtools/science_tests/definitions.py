"""Load and validate release science-test definitions."""

import csv
import fnmatch
import hashlib
import json
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path

from simtools.constants import RUN_TIME_ENVIRONMENT_SCHEMA, SCIENCE_TEST_SCHEMA_PATH
from simtools.data_model import schema
from simtools.io import ascii_handler
from simtools.production_configuration.job_grid_io import read_job_grid
from simtools.production_configuration.production_file_selection import (
    discover_manifests,
    normalize_for_comparison,
)
from simtools.utils.general import replace_placeholders_recursively

_SCIENCE_REPORT_PATH = "__SCIENCE_REPORT_PATH__"
_JOB_IDS = "job_ids"

_ARRAY_LAYOUT_KEY = "array_layout"
_COMPARISON_INVENTORY_FILE = "comparison_inventory.json"
_REASON_KEY = "reason"
_SCOPE_KEY = "scope"
_SELECT_KEY = "select"
_STATUS_KEY = "status"
_TELESCOPE_TYPE_KEY = "telescope_type"
_ENCODING = "utf-8"
_CSV_CONTEXT_KEYS = ("pair", _ARRAY_LAYOUT_KEY, _SCOPE_KEY, _TELESCOPE_TYPE_KEY)

_VALID_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")

_PATH_KEYS = (
    "__SCIENCE_CANDIDATE_PATH__",
    "__SCIENCE_BASELINE_PATH__",
    "__PRODUCTION_CONFIGURATION_PATH__",
)

_MODEL_SOURCE_KEYS = (
    "simulation_models_path",
    "simulation_models_git_path",
    "simulation_models_git_revision",
)

_SUBMISSION_FILE_NAME = "submission.json"

_ACCEPTANCE_VALUE_FORMAT = ".4g"

_REPORT_PAGE_COLUMN = "report_page"

_SITE_SCHEMA_FILE = "site.schema.yml"
_RELEASE_FILE_NAME = "release.yml"

_SETUP_RELEASE_LABEL = "__SCIENCE_RELEASE_LABEL__"

_SCIENCE_TESTS_ROOT_ENV = "SIMTOOLS_SCIENCE_TESTS_ROOT"

_PUBLICATION_WARNING_BYTES = 20 * 1024 * 1024

_PUBLICATION_WARNING_FILES = 30
_EXECUTION_LOG_FILE_NAME = "execution.log"
_WORKFLOW_FILE_NAME = "workflow.yml"

logger = logging.getLogger(__name__)


class _AcceptanceError(ValueError):
    """A valid measurement exceeds a configured acceptance limit."""


def _validate_release_context(release, context, require_baseline=True):
    """Validate the release label and external directory paths."""
    label = release.get("release_label")
    if not isinstance(label, str) or not _VALID_NAME.fullmatch(label):
        raise ValueError("release_label must be a non-empty filename-safe label.")
    required_keys = _PATH_KEYS if require_baseline else (_PATH_KEYS[0], _PATH_KEYS[2])
    for key in required_keys:
        value = context.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute() or "__" in value:
            raise ValueError(f"Context requires an absolute path for {key}.")
    candidate = Path(context[_PATH_KEYS[0]]).resolve()
    baseline = Path(context[_PATH_KEYS[1]]).resolve() if require_baseline else None
    if require_baseline and (
        candidate == baseline or candidate in baseline.parents or baseline in candidate.parents
    ):
        raise ValueError(
            "Candidate and baseline paths must be separate, non-overlapping directories: "
            f"{_PATH_KEYS[0]}={candidate}; {_PATH_KEYS[1]}={baseline}."
        )


def _select(requested, available, kind):
    """Validate selections, matching science-test sites without regard to case."""
    if kind == "site" and requested is not None:
        site_names = {site.casefold(): site for site in available}
        requested = [site_names.get(site.casefold(), site) for site in requested]
    selected = list(dict.fromkeys(requested if requested is not None else available))
    if not selected or set(selected) - set(available):
        raise ValueError(f"Invalid {kind} selection: {selected}; available: {sorted(available)}")
    return selected


def _comparison_selected(requested_sites, requested_tests, suites, catalogue):
    """Return whether the selected test dependency graph needs two productions."""
    selected_test_ids = set()
    for site in requested_sites:
        selected = suites[site]["required_tests"] if requested_tests is None else requested_tests
        selected_test_ids.update(_dependency_order(selected, catalogue))
    return any(test_id.startswith("compare.") for test_id in selected_test_ids)


def _validate_selected_context(
    release, context, requested_sites, requested_tests, suites, catalogue
):
    """Validate context paths for the selected test dependency graph."""
    comparison_selected = _comparison_selected(requested_sites, requested_tests, suites, catalogue)
    _validate_release_context(release, context, require_baseline=comparison_selected)
    if not comparison_selected and _PATH_KEYS[1] in context:
        candidate = Path(context[_PATH_KEYS[0]]).resolve()
        baseline = Path(context[_PATH_KEYS[1]]).resolve()
        if candidate == baseline:
            logger.warning(
                "Candidate and baseline paths are identical; this is allowed for "
                "non-comparison science tests."
            )


def _load_suites(release_dir, release, catalogue, template_dir=None):
    """Read every required suite before any selected test is executed."""
    sites = release.get("required_sites")
    if not isinstance(sites, list) or not sites or len(sites) != len(set(sites)):
        raise ValueError("required_sites must be a non-empty unique list.")
    suites = {}
    for site in sites:
        if not isinstance(site, str) or not _VALID_NAME.fullmatch(site):
            raise ValueError(f"Invalid site name: {site!r}")
        suite = _load_site_suite(release_dir, template_dir, site)
        _validate_science_test_config(suite, _SITE_SCHEMA_FILE)
        _validate_site_suite(site, suite, catalogue)
        suites[site] = suite
    return suites


def _load_site_suite(release_dir, template_dir, site):
    """Load shared site defaults and an optional release-specific override."""
    site_file = release_dir / "sites" / f"{site}.yml"
    suite = {}
    if template_dir:
        defaults = template_dir / "sites" / f"{site}.yml"
        if defaults.is_file():
            suite.update(_load_yaml(defaults))
    if site_file.is_file():
        suite.update(_load_yaml(site_file))
    if not suite:
        raise FileNotFoundError(site_file)
    return suite


def _validate_site_suite(site, suite, catalogue):
    """Validate one site's layout and required-test declarations."""
    required = suite.get("required_tests")
    if suite.get("site") != site or not suite.get("array_layout_name"):
        raise ValueError(f"Invalid site or missing layout in suite {site}.")
    if not isinstance(required, list) or not required or len(required) != len(set(required)):
        raise ValueError(f"Suite {site} needs a non-empty unique required_tests list.")
    for test_id in _dependency_order(required, catalogue):
        _validate_definition(test_id, site, catalogue[test_id])


def _validate_definition(test_id, site, definition):
    """Check a test's identity, site support, cost tier, and workflow."""
    if not isinstance(test_id, str) or not _VALID_NAME.fullmatch(test_id):
        raise ValueError(f"Invalid science-test ID: {test_id!r}")
    if site not in definition.get("sites", []):
        raise ValueError(f"Test {test_id!r} does not support site {site!r}")
    if definition.get("tier") not in {"smoke", "release-light", "release-prod"}:
        raise ValueError(f"Invalid tier for {test_id}.")
    if not definition.get("workflow"):
        raise ValueError(f"Missing workflow for {test_id}.")


def _check_production_selection(test_id, definition, tests, allow_production, dry_run):
    """Require production to be explicitly selected after grid review."""
    if not dry_run and definition["tier"] == "release-prod":
        if not allow_production or tests is None or test_id not in tests:
            raise ValueError(
                "Select production explicitly with --test and --allow_production "
                "after reviewing the generated grid."
            )


def _matched_files(pattern):
    """Resolve a scoped pattern and require non-empty evidence."""
    path = Path(pattern)
    root = Path(path.anchor or ".")
    relative = str(path.relative_to(root)) if path.is_absolute() else pattern
    files = sorted(p for p in root.glob(relative) if p.is_file())
    if not files or any(path.stat().st_size == 0 for path in files):
        raise ValueError(f"Missing or empty required product: {pattern}")
    return files


def _check_inputs(definition, replacements):
    """Require reviewed inputs and durable evidence of completed productions."""
    for pattern in definition.get("requires", []):
        _matched_files(replace_placeholders_recursively(pattern, replacements))
    if definition.get("produces_production"):
        root = Path(replacements["__SCIENCE_CANDIDATE_SITE_PATH__"])
        submission = next(root.rglob(_SUBMISSION_FILE_NAME), None)
        if submission is not None:
            raise ValueError(
                f"Production submission blocked: existing record {submission}. "
                "This also blocks retries of failed jobs; --allow_production does not override it. "
                "Use --overwrite to archive and retry failed production once all jobs have ended, "
                "or use a new candidate path."
            )
    if definition.get("reviewed_grid") and not definition.get("collect_production"):
        _check_reviewed_production(definition, replacements)
        if definition.get("acceptance_rule"):
            baseline = replacements | {
                "__SCIENCE_CANDIDATE_PATH__": replacements["__SCIENCE_BASELINE_PATH__"],
                "__SCIENCE_CANDIDATE_SITE_PATH__": replacements["__SCIENCE_BASELINE_SITE_PATH__"],
            }
            _check_reviewed_production(definition, baseline)
    if definition.get("requires_completed_production"):
        root = Path(replacements["__SCIENCE_CANDIDATE_SITE_PATH__"])
        for path in _matched_files(str(root / "**" / _SUBMISSION_FILE_NAME)):
            _check_completed_submission(path)


def _check_completed_submission(path):
    """Require a completed manifest covering every submitted job's declared files."""
    payload = json.loads(path.read_text(encoding=_ENCODING))
    metadata = payload.get("metadata", {})
    if metadata.get("state") != "completed":
        raise ValueError(f"Production is not completed: {path}")
    jobs = payload.get(_JOB_IDS, [])
    outputs = metadata.get("expected_outputs", {})
    if not jobs or set(jobs) != set(outputs):
        raise ValueError(f"Production has missing expected outputs: {path}")
    if any(not paths for paths in outputs.values()) or any(
        not _nonempty_output(Path(p)) for paths in outputs.values() for p in paths
    ):
        raise ValueError(f"Production has missing expected outputs: {path}")


def _evaluate_products(definition, replacements):
    """Enforce artifact completeness and declared numerical limits."""
    _check_required_products(definition, replacements)
    _check_expected_pair_count(definition, replacements)
    rule = definition.get("rule", {})
    warnings = []
    for metric in rule.get("metrics", []):
        try:
            warnings.extend(_evaluate_metric(metric, replacements))
        except _AcceptanceError as exc:
            warnings.append(str(exc))
    if rule.get("mode") == "advisory":
        warnings.append("Numerical acceptance is advisory; review required.")
    if warnings:
        warnings = list(dict.fromkeys(warnings))
        return {
            _STATUS_KEY: "warn",
            _REASON_KEY: f"Acceptance warning ({len(warnings)}): {warnings[0]}",
            "acceptance_warnings": warnings,
        }
    return {_STATUS_KEY: "pass"}


def _evaluate_metric(metric, replacements):
    """Evaluate one configured metric over all matching products."""
    pattern = replace_placeholders_recursively(metric["file"], replacements)
    warnings = []
    for path in _matched_files(pattern):
        if metric.get("format") == "csv":
            warnings.extend(_evaluate_csv_metric(path, metric))
        else:
            _evaluate_json_metric(path, metric)
    return warnings


def _evaluate_json_metric(path, metric):
    """Read and validate one nested JSON metric."""
    value = json.loads(path.read_text(encoding=_ENCODING))
    try:
        for key in metric["key"].split("."):
            value = value[int(key)] if isinstance(value, list) else value[key]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Missing metric {metric['key']} in {path}") from exc
    _check_metric(value, metric)


def _evaluate_csv_metric(path, metric):
    """Apply configured scalar thresholds to selected rows of a compact results table."""
    field = metric.get("field", metric["key"])
    with path.open(encoding=_ENCODING, newline="") as input_file:
        all_rows = list(csv.DictReader(input_file))
    selectors = metric.get(_SELECT_KEY, {})
    rows = [row for row in all_rows if _csv_row_matches(row, selectors)]
    counts = Counter(_csv_metric_identity(row) for row in all_rows)
    duplicates = [identity for identity, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate comparison metric rows in {path}: {duplicates[:5]}.")
    warnings = _missing_csv_metrics(path, all_rows, rows, selectors)
    if not rows:
        return warnings or [f"No comparison rows match {selectors} in {path}."]
    for row in rows:
        try:
            warning = _evaluate_csv_row(row, metric, field, path)
        except _AcceptanceError as exc:
            warning = str(exc)
        if warning:
            warnings.append(warning)
    return warnings


def _csv_row_matches(row, selectors):
    """Apply configured selectors to a scalar result or expected context."""
    return all(_csv_value_matches(row.get(key, ""), value) for key, value in selectors.items())


def _csv_metric_identity(row):
    """Identify a scalar measurement independently of its value and PDF page."""
    return tuple(
        row.get(key, "")
        for key in (
            "pair",
            _ARRAY_LAYOUT_KEY,
            "plot",
            _SCOPE_KEY,
            _TELESCOPE_TYPE_KEY,
            "population",
            "metric",
        )
    )


def _csv_metric_contexts(path, rows):
    """Use declared pair/type coverage, falling back to older scalar tables."""
    inventory = path.with_name(_COMPARISON_INVENTORY_FILE)
    payload = json.loads(inventory.read_text(encoding=_ENCODING)) if inventory.is_file() else {}
    scopes = payload.get("metric_scopes")
    if scopes is None:
        return {tuple(row.get(key, "") for key in _CSV_CONTEXT_KEYS) for row in rows}
    return {
        (pair, layout, _TELESCOPE_TYPE_KEY if telescope else "whole_array", telescope)
        for pair, layouts in scopes.items()
        for layout, telescopes in layouts.items()
        for telescope in ("", *telescopes)
    }


def _missing_csv_metrics(path, all_rows, selected, selectors):
    """Warn for each expected configuration/type without a selected measurement."""
    observed = {tuple(row.get(key, "") for key in _CSV_CONTEXT_KEYS) for row in selected}
    context_selectors = {key: value for key, value in selectors.items() if key in _CSV_CONTEXT_KEYS}
    warnings = []
    for context in sorted(_csv_metric_contexts(path, all_rows) - observed):
        row = dict(zip(_CSV_CONTEXT_KEYS, context))
        if not _csv_row_matches(row, context_selectors):
            continue
        telescope = row[_TELESCOPE_TYPE_KEY]
        plots = selectors.get("plot", [])
        plots = [plots] if isinstance(plots, str) else plots
        if (
            telescope
            and plots
            and not any(fnmatch.fnmatchcase(telescope, plot.rsplit("_", 1)[-1]) for plot in plots)
        ):
            continue
        warnings.append(
            f"Missing comparison metric rows for {row['pair']} {row[_ARRAY_LAYOUT_KEY]} "
            f"{row[_SCOPE_KEY]} {telescope}: {selectors}."
        )
    return warnings


def _evaluate_csv_row(row, metric, field, path):
    """Evaluate a selected CSV row and return its warning, if any."""
    label = f"{row.get('plot')} {row.get(_SCOPE_KEY)} {row.get('pair')}"
    if row.get("valid", "").casefold() != "true":
        return f"{label}: invalid statistic ({row.get(_REASON_KEY)}){_report_page_suffix(row)}."
    value, counts = _read_csv_acceptance_values(row, field, path)
    minimum_entries = metric.get("minimum_entries")
    if minimum_entries is not None and min(counts) < minimum_entries:
        return f"{label}: insufficient statistics ({counts}){_report_page_suffix(row)}."
    return _check_csv_thresholds(value, metric, field, label, row)


def _read_csv_acceptance_values(row, field, path):
    """Parse and validate the statistic and sample counts from a CSV row."""
    try:
        value = float(row[field])
        counts = [int(float(row[key])) for key in ("baseline_count", "candidate_count")]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid acceptance value in {path}: {row}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Invalid acceptance value in {path}: {row}")
    return value, counts


def _check_csv_thresholds(value, metric, field, label, row):
    """Apply warning and failure thresholds to one valid statistic."""
    minimum = metric.get("minimum", -math.inf)
    if value < minimum:
        raise _AcceptanceError(
            f"{label}: {field}={_format_acceptance_value(value)} is below required minimum "
            f"{_format_acceptance_value(minimum)}{_report_page_suffix(row)}."
        )
    warning_limit = metric.get("warning_maximum")
    fail_limit = metric.get("maximum", math.inf)
    if value > fail_limit:
        raise _AcceptanceError(
            f"Acceptance failed for {label}: {field}={_format_acceptance_value(value)} "
            f"exceeds {_format_acceptance_value(fail_limit)}{_report_page_suffix(row)}."
        )
    if warning_limit is not None and value > warning_limit:
        return (
            f"{label}: {field}={_format_acceptance_value(value)} exceeds warning limit "
            f"{_format_acceptance_value(warning_limit)}{_report_page_suffix(row)}."
        )
    return None


def _report_page_suffix(row):
    """Return a compact PDF page reference when the statistic was rendered."""
    page = row.get(_REPORT_PAGE_COLUMN)
    return f" (PDF page {page})" if page else ""


def _format_acceptance_value(value):
    """Render acceptance values to four significant digits for readable reports."""
    return format(value, _ACCEPTANCE_VALUE_FORMAT)


def _csv_value_matches(actual, expected):
    """Match an exact CSV selector or one of its configured glob patterns."""
    choices = expected if isinstance(expected, list) else [expected]
    return any(fnmatch.fnmatchcase(actual, str(choice)) for choice in choices)


def _check_required_products(definition, replacements):
    """Require declared products, allowing the consolidated comparison summary."""
    for pattern in definition.get("products", []):
        resolved_pattern = replace_placeholders_recursively(pattern, replacements)
        _matched_files(resolved_pattern)


def _check_expected_pair_count(definition, replacements):
    """Require all reviewed trigger-comparison configurations to be represented."""
    expected = definition.get("expected_pair_count")
    if expected is None:
        return
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
        raise ValueError("expected_pair_count must be a positive integer.")
    inventory = Path(replacements[_SCIENCE_REPORT_PATH]) / _COMPARISON_INVENTORY_FILE
    payload = json.loads(inventory.read_text(encoding=_ENCODING))
    observed = payload.get("matched_pair_count")
    if observed != expected:
        raise ValueError(
            f"Incomplete comparison matrix: expected {expected} matched pairs, observed {observed}."
        )


def _check_metric(value, metric):
    """Separate invalid measurements from valid values outside an allowance."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"Invalid metric {metric['key']}: {value}")
    if value < metric.get("minimum", -math.inf) or value > metric.get("maximum", math.inf):
        raise _AcceptanceError(
            f"Acceptance failed for {metric['key']}: {_format_acceptance_value(value)}"
        )


def _campaign_signature(
    release_dir,
    template_dir,
    release,
    catalogue,
    context,
    suites,
    products,
    rules,
    model_source,
):
    """Identify definitions and external context without publishing absolute paths."""
    payload = [
        release,
        catalogue,
        context,
        suites,
        products,
        rules,
        model_source,
    ]
    for definition in catalogue.values():
        payload.append(
            _resolve_workflow(template_dir, release_dir, definition["workflow"]).read_text(
                encoding=_ENCODING
            )
        )
    runtime_file = template_dir / "run_time.yml"
    if runtime_file.is_file():
        payload.append(runtime_file.read_text(encoding=_ENCODING))
    for profile in sorted((template_dir / "profiles").glob("*.yml")):
        payload.append(profile.read_text(encoding=_ENCODING))
    for environment_file in sorted((template_dir / "profiles").glob("*.env")):
        payload.append(environment_file.read_text(encoding=_ENCODING))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode(_ENCODING)).hexdigest()


def _load_acceptance(release_dir, template_dir):
    """Load the shared acceptance contracts and reject unsupported overrides."""
    product_contract = _load_yaml(template_dir / "acceptance" / "expected-products.yml")
    threshold_contract = _load_yaml(template_dir / "acceptance" / "thresholds.yml")
    _validate_science_test_config(product_contract, "expected_products.schema.yml")
    _validate_science_test_config(threshold_contract, "thresholds.schema.yml")
    products = product_contract.get("tests")
    rules = threshold_contract.get("rules")
    if not isinstance(products, dict) or not isinstance(rules, dict):
        raise ValueError("Expected products and acceptance rules must be mappings.")
    for rule_id, rule in rules.items():
        _validate_rule(rule, rule_id)
    overrides = release_dir / "acceptance" / "overrides.yml"
    if overrides.is_file() and _load_yaml(overrides).get("overrides"):
        raise ValueError(
            "Release acceptance overrides are not supported yet; review them explicitly."
        )
    return {"products": products, "rules": rules}


def _test_definition(test_id, catalogue, acceptance, release):
    """Attach one test's shared artifact and acceptance contracts."""
    products = acceptance["products"].get(test_id)
    if (
        not isinstance(products, list)
        or not products
        or not all(isinstance(p, str) for p in products)
    ):
        raise ValueError(f"Missing expected-product contract for {test_id}.")
    rule_id = catalogue[test_id].get("acceptance_rule")
    if rule_id and rule_id not in acceptance["rules"]:
        raise ValueError(f"Missing acceptance rule {rule_id} for {test_id}.")
    rule = acceptance["rules"].get(rule_id, {})
    _validate_rule(rule, rule_id)
    definition = {
        **catalogue[test_id],
        "products": products,
        "rule": rule,
        "expected_changes": release.get("expected_changes", []),
        "baseline": release.get("baseline"),
        "release_label": release.get("release_label"),
        "model_version": release.get("model_version"),
        "report_mode": release.get("report_mode", "compact"),
    }
    comparison = test_id.removeprefix("compare.")
    requirements = _comparison_settings(release, comparison, "requirements")
    parameters = _comparison_settings(release, comparison, "parameters")
    if requirements:
        definition.update(requirements)
    if parameters:
        definition["comparison_parameters"] = parameters
    return definition


def _comparison_settings(release, comparison, kind):
    """Read one comparison's release overrides, validating their mapping structure."""
    key = f"comparison_{kind}"
    settings = release.get(key, {})
    if not isinstance(settings, dict):
        raise ValueError(f"{key} must be a mapping.")
    overrides = settings.get(comparison, {})
    if overrides and not isinstance(overrides, dict):
        raise ValueError(f"Invalid comparison {kind} for {comparison}.")
    return overrides


def _validate_rule(rule, rule_id):
    """Fail malformed acceptance rules before any workflow can execute."""
    if not isinstance(rule, dict) or (
        rule_id and rule.get("mode") not in {"advisory", "blocking", "thresholds"}
    ):
        raise ValueError(f"Invalid acceptance rule: {rule_id}")
    metrics = rule.get("metrics", [])
    if not isinstance(metrics, list):
        raise ValueError("Acceptance metrics must be a list.")
    for metric in metrics:
        _validate_metric_rule(metric)


def _validate_metric_rule(metric):
    """Validate a metric's JSON selector and finite numerical limits."""
    if (
        not isinstance(metric, dict)
        or not isinstance(metric.get("file"), str)
        or not isinstance(metric.get("key"), str)
    ):
        raise ValueError("Each acceptance metric needs file and key.")
    for limit in ("minimum", "maximum", "warning_maximum"):
        if limit not in metric:
            continue
        value = metric[limit]
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"Invalid acceptance {limit}: {value}")
    selectors = metric.get(_SELECT_KEY, {})
    if not isinstance(selectors, dict) or any(
        not isinstance(key, str)
        or not isinstance(value, str | list)
        or (isinstance(value, list) and not all(isinstance(item, str) for item in value))
        for key, value in selectors.items()
    ):
        raise ValueError("CSV acceptance select must map column names to string values.")
    minimum_entries = metric.get("minimum_entries")
    if minimum_entries is not None and (
        isinstance(minimum_entries, bool)
        or not isinstance(minimum_entries, int)
        or minimum_entries < 1
    ):
        raise ValueError(f"Invalid acceptance minimum_entries: {minimum_entries}")


def _model_source_arguments(application_args, context, release):
    """Apply campaign model settings over runner defaults."""
    args = dict(application_args or {})
    for configuration in (context, release):
        args.update({key: configuration[key] for key in _MODEL_SOURCE_KEYS if key in configuration})
    source = next(
        (
            configuration
            for configuration in (release, context, args)
            if configuration.get("simulation_models_git_path")
            or configuration.get("simulation_models_path")
        ),
        args,
    )
    if source.get("simulation_models_git_path"):
        args["simulation_models_path"] = None
    elif source.get("simulation_models_path"):
        args["simulation_models_git_path"] = None
        args["simulation_models_git_revision"] = None
    return args


def _shared_runtime_environment(template_dir, replacements):
    """Load shared runtime settings with paths relative to the runtime file."""
    runtime_file = template_dir / "run_time.yml"
    if not runtime_file.is_file():
        return None
    runtime_replacements = dict(replacements)
    runtime_replacements["__CONFIG_DIRECTORY__"] = str(runtime_file.parent.resolve())
    configuration = replace_placeholders_recursively(_load_yaml(runtime_file), runtime_replacements)
    schema.validate_dict_using_schema(
        configuration, schema_file=RUN_TIME_ENVIRONMENT_SCHEMA, offline=True
    )
    return configuration["runtime_environment"]


def _site_replacements(context, site, site_config, work):
    """Build placeholder values without requiring site-specific workflows."""
    site_label = site_config.get("site_label", site.title())
    candidate_root = Path(context.get("__SCIENCE_CANDIDATE_PATH__", ""))
    baseline_root = Path(context.get("__SCIENCE_BASELINE_PATH__", ""))
    replacements = dict(context)
    replacements.update(
        {
            "__SCIENCE_SITE_KEY__": site,
            "__SCIENCE_SITE__": site_label,
            "__SCIENCE_ARRAY_LAYOUT__": site_config["array_layout_name"],
            "__SCIENCE_CANDIDATE_SITE_PATH__": str(candidate_root / site),
            "__SCIENCE_BASELINE_SITE_PATH__": str(baseline_root / site),
            _SCIENCE_REPORT_PATH: str(work),
            "__SCIENCE_COLLECTION_PATH__": str(work),
        }
    )
    for key, value in site_config.get("replacements", {}).items():
        if key in replacements:
            raise ValueError(f"Site replacements cannot override reserved context key: {key}")
        replacements[key] = value
    return {str(key): str(value) for key, value in replacements.items()}


def _dependency_order(test_ids, catalogue):
    """Return selected tests once each, with dependencies first."""
    ordered = []
    visiting = set()

    def visit(test_id):
        if test_id in ordered:
            return
        if test_id in visiting:
            raise ValueError(f"Circular science-test dependency involving {test_id!r}")
        if test_id not in catalogue:
            raise ValueError(f"Unknown science test: {test_id!r}")
        if not isinstance(catalogue[test_id], dict):
            raise ValueError(f"Invalid definition for science test: {test_id!r}")
        visiting.add(test_id)
        for dependency in catalogue[test_id].get("depends_on", ()):
            visit(dependency)
        visiting.remove(test_id)
        ordered.append(test_id)

    for test_id in test_ids:
        visit(test_id)
    return ordered


def _load_catalogue(release, release_dir, template_dir):
    path = Path(release["catalogue"]) if "catalogue" in release else template_dir / "catalogue.yml"
    if not path.is_absolute():
        path = (release_dir / path).resolve()
    data = _load_yaml(path)
    _validate_science_test_config(data, "catalogue.schema.yml")
    catalogue = data.get("tests", data)
    if not isinstance(catalogue, dict) or not catalogue:
        raise ValueError("Science-test catalogue must be a non-empty mapping.")
    return catalogue


def _resolve_template_dir(release_dir, template_dir):
    if template_dir is not None:
        return Path(template_dir).resolve()
    return template_dir_for_release(release_dir)


def template_dir_for_release(release_dir):
    """Find the template above a release bundle.

    Parameters
    ----------
    release_dir : pathlib.Path
        Release directory whose ancestors are searched.

    Returns
    -------
    pathlib.Path
        Shared science-test template directory.

    Raises
    ------
    FileNotFoundError
        No ancestor contains the shared template.
    """
    for parent in release_dir.parents:
        candidate = parent / "science-test-template"
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not find science-test-template beside release bundle.")


def _resolve_workflow(template_dir, release_dir, workflow):
    path = Path(workflow)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Workflow must be a relative template path: {workflow}")
    release_override = release_dir / path
    return release_override if release_override.is_file() else template_dir / path


def _relative_path(path, root):
    """Return a stable path for reports without exposing a checkout location."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path.name)


def _load_context(context_file):
    """Load context paths, resolving the optional shared storage root."""
    if context_file is None:
        return {}
    context = _load_yaml(Path(context_file))
    _validate_science_test_config(context, "context.schema.yml")
    placeholder = f"${{{_SCIENCE_TESTS_ROOT_ENV}}}"
    if not any(isinstance(value, str) and placeholder in value for value in context.values()):
        return context
    storage_root = os.environ.get(_SCIENCE_TESTS_ROOT_ENV)
    if not storage_root or not Path(storage_root).is_absolute():
        raise ValueError(
            f"Set {_SCIENCE_TESTS_ROOT_ENV} to a non-empty absolute path "
            f"before loading {context_file}."
        )
    return replace_placeholders_recursively(context, {placeholder: str(Path(storage_root))})


def _validate_science_test_config(configuration, schema_file):
    """Validate one science-test configuration against its packaged schema."""
    schema.validate_dict_using_schema(
        configuration,
        schema_file=SCIENCE_TEST_SCHEMA_PATH / schema_file,
        offline=True,
    )


def _load_yaml(path):
    if not path.exists():
        raise FileNotFoundError(path)
    data = ascii_handler.collect_data_from_file(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def _nonempty_output(path):
    """Require a nonempty file or a populated job directory."""
    if path.is_file():
        return path.stat().st_size > 0
    return path.is_dir() and any(p.is_file() and p.stat().st_size > 0 for p in path.rglob("*"))


def _check_reviewed_production(definition, replacements):
    """Reconcile completed jobs and their configurations with the reviewed grid."""
    grid = replace_placeholders_recursively(definition["reviewed_grid"], replacements)
    rows, metadata = read_job_grid(grid)
    root = Path(replacements["__SCIENCE_CANDIDATE_SITE_PATH__"])
    manifests = discover_manifests(root)
    expected_ids = {f"job-{index + 1:06d}" for index in range(len(rows))}
    observed_ids = [manifest.data.get("job_id") for manifest in manifests]
    _require_same_ids(expected_ids, observed_ids, "production jobs")
    by_id = {manifest.data["job_id"]: manifest for manifest in manifests}
    for index, row in enumerate(rows):
        manifest = by_id[f"job-{index + 1:06d}"]
        _check_grid_configuration(row, metadata, manifest.data["configuration"])
        files = manifest.data["files"]
        if not files.get("reduced_event_data"):
            raise ValueError(f"Missing reduced event data in {manifest.path}.")
        for paths in files.values():
            for relative in paths:
                path = (manifest.directory / relative).resolve()
                if not path.is_relative_to(manifest.directory.resolve()) or not _nonempty_output(
                    path
                ):
                    raise ValueError(f"Missing or empty production output: {path}")
    submitted = []
    for path in _matched_files(str(root / "**" / _SUBMISSION_FILE_NAME)):
        _check_completed_submission(path)
        submitted.extend(json.loads(path.read_text(encoding=_ENCODING)).get(_JOB_IDS, []))
    _require_same_ids(expected_ids, submitted, "submitted jobs")
    return {"expected_jobs": len(rows), "observed_jobs": len(manifests)}


def _require_same_ids(expected, observed, description):
    """Report bounded missing/extra diagnostics and reject duplicate identities."""
    counts = Counter(observed)
    missing = expected - counts.keys()
    extra = counts.keys() - expected
    duplicate = [key for key, count in counts.items() if count > 1]
    if missing or extra or duplicate:
        raise ValueError(
            f"Incomplete {description}: missing={sorted(missing)[:5]}, "
            f"extra={sorted(extra)[:5]}, duplicate={duplicate[:5]}."
        )


def _check_grid_configuration(row, metadata, configuration):
    """Check simulation-defining grid fields, including each run number."""
    excluded = {
        "nsb_rate",
        "overwrite_model_parameters",
        "scan_label",
        "model_parameter_set",
    }
    for key, value in (
        row | {key: metadata[key] for key in ("site", "simulation_software") if key in metadata}
    ).items():
        if key in excluded:
            continue
        observed = configuration.get(key)
        if (
            key in {"model_version", "array_layout_name"}
            and isinstance(observed, list)
            and len(observed) == 1
        ):
            observed = observed[0]
        if normalize_for_comparison(value) != normalize_for_comparison(observed):
            raise ValueError(
                f"Reviewed grid mismatch for run {row['run_number']}, {key}: "
                f"expected {value}, observed {observed}."
            )

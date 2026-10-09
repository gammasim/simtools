"""Small runner for reusable, site-parameterised release science tests."""

import hashlib
import json
import logging
import math
import re
import shutil
import subprocess
from pathlib import Path

from simtools.constants import RUN_TIME_ENVIRONMENT_SCHEMA, SCHEMA_PATH
from simtools.data_model import schema
from simtools.io import ascii_handler
from simtools.job_execution.backends.registry import get_backend
from simtools.job_execution.execution import collect_submission, load_submission
from simtools.job_execution.job_manager import JobExecutionError
from simtools.runners.simtools_runner import prepare_workflow, run_applications
from simtools.utils.general import get_uuid, replace_placeholders_recursively

logger = logging.getLogger(__name__)

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
_RELEASE_FILE_NAME = "release.yml"
_SETUP_RELEASE_LABEL = "__SCIENCE_RELEASE_LABEL__"


class _AcceptanceError(ValueError):
    """A valid measurement exceeds a configured acceptance limit."""


def _validate_release_context(release, context, require_distinct_paths=True):
    """Validate the release label and external directory paths."""
    label = release.get("release_label")
    if not isinstance(label, str) or not _VALID_NAME.fullmatch(label):
        raise ValueError("release_label must be a non-empty filename-safe label.")
    for key in _PATH_KEYS:
        value = context.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute() or "__" in value:
            raise ValueError(f"Context requires an absolute path for {key}.")
    candidate = Path(context[_PATH_KEYS[0]]).resolve()
    baseline = Path(context[_PATH_KEYS[1]]).resolve()
    if require_distinct_paths and candidate == baseline:
        raise ValueError(
            "Candidate and baseline paths must differ: "
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


def _warn_on_identical_paths(context, comparison_selected):
    """Warn when a non-comparison run uses one path for both productions."""
    if comparison_selected:
        return
    candidate = Path(context[_PATH_KEYS[0]]).resolve()
    baseline = Path(context[_PATH_KEYS[1]]).resolve()
    if candidate == baseline:
        logger.warning(
            "Candidate and baseline paths are identical; this is allowed for "
            "non-comparison science tests."
        )


def _validate_selected_context(
    release, context, requested_sites, requested_tests, suites, catalogue
):
    """Validate context paths for the selected test dependency graph."""
    comparison_selected = _comparison_selected(requested_sites, requested_tests, suites, catalogue)
    _validate_release_context(release, context, require_distinct_paths=comparison_selected)
    _warn_on_identical_paths(context, comparison_selected)


def _load_suites(release_dir, release, catalogue):
    """Read every required suite before any selected test is executed."""
    sites = release.get("required_sites")
    if not isinstance(sites, list) or not sites or len(sites) != len(set(sites)):
        raise ValueError("required_sites must be a non-empty unique list.")
    suites = {}
    for site in sites:
        if not isinstance(site, str) or not _VALID_NAME.fullmatch(site):
            raise ValueError(f"Invalid site name: {site!r}")
        suite = _load_yaml(release_dir / "sites" / f"{site}.yml")
        required = suite.get("required_tests")
        if suite.get("site") != site or not suite.get("array_layout_name"):
            raise ValueError(f"Invalid site or missing layout in suite {site}.")
        if not isinstance(required, list) or not required or len(required) != len(set(required)):
            raise ValueError(f"Suite {site} needs a non-empty unique required_tests list.")
        for test_id in _dependency_order(required, catalogue):
            _validate_definition(test_id, site, catalogue[test_id])
        suites[site] = suite
    return suites


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


def _execute_selection(prepared, dry_run, run_id, application_args):
    """Execute dependency order and record blocked downstream tests."""
    results = []
    for args, planned in prepared:
        if dry_run:
            _log_test_execution(planned, "Planned")
            results.append(planned)
            continue
        dependencies = {f"{name}.{planned['site']}" for name in args[-1].get("depends_on", [])}
        if any(r["id"] in dependencies and r["status"] not in {"pass", "warn"} for r in results):
            results.append(
                {
                    **planned,
                    "status": _dependency_status(results, dependencies),
                    "execution": _dependency_status(results, dependencies),
                    "reason": "Dependency failed or is not yet complete.",
                }
            )
        else:
            _log_test_execution(planned, "Running")
            results.append(
                _run_test(*args, dry_run=False, run_id=run_id, application_args=application_args)
            )
    return results


def _dependency_status(results, dependencies):
    """Distinguish queued dependencies from failed dependencies."""
    if all(
        r["status"] in {"pass", "warn", "submitted", "pending"}
        for r in results
        if r["id"] in dependencies
    ):
        return "pending"
    return "blocked"


def _log_test_execution(result, action):
    """Show the test, site, runtime, and application output paths once."""
    logger.info("%s science test: %s", action, result["id"])
    logger.info("  Runtime: %s", result["runtime"])
    logger.info("  Output: %s", ", ".join(result["output_paths"]))


def _runtime_description(runtime, ignored):
    """Describe the effective runtime without starting a container."""
    if ignored or runtime is None:
        return "host (no container)"
    return f"{runtime.get('container_engine', 'docker')} (image: {runtime['image']})"


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
    if definition.get("requires_completed_production"):
        root = Path(replacements["__SCIENCE_CANDIDATE_SITE_PATH__"])
        for path in _matched_files(str(root / "**" / _SUBMISSION_FILE_NAME)):
            _check_completed_submission(path)


def _check_completed_submission(path):
    """Require a completed manifest covering every submitted job's declared files."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    if metadata.get("state") != "completed":
        raise ValueError(f"Production is not completed: {path}")
    jobs = payload.get("job_ids", [])
    outputs = metadata.get("expected_outputs", {})
    if not jobs or set(jobs) != set(outputs):
        raise ValueError(f"Production has missing expected outputs: {path}")
    if any(not paths for paths in outputs.values()) or any(
        not Path(p).exists() for paths in outputs.values() for p in paths
    ):
        raise ValueError(f"Production has missing expected outputs: {path}")


def _evaluate_products(definition, replacements):
    """Enforce artifact completeness and declared numerical limits."""
    for pattern in definition.get("products", []):
        _matched_files(replace_placeholders_recursively(pattern, replacements))
    rule = definition.get("rule", {})
    for metric in rule.get("metrics", []):
        pattern = replace_placeholders_recursively(metric["file"], replacements)
        for path in _matched_files(pattern):
            value = json.loads(path.read_text(encoding="utf-8"))
            try:
                for key in metric["key"].split("."):
                    value = value[int(key)] if isinstance(value, list) else value[key]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError(f"Missing metric {metric['key']} in {path}") from exc
            _check_metric(value, metric)
    if rule.get("mode") == "advisory":
        return {"status": "warn", "reason": "Numerical acceptance is advisory; review required."}
    return {"status": "pass"}


def _check_metric(value, metric):
    """Separate invalid measurements from valid values outside an allowance."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"Invalid metric {metric['key']}: {value}")
    if value < metric.get("minimum", -math.inf) or value > metric.get("maximum", math.inf):
        raise _AcceptanceError(f"Acceptance failed for {metric['key']}: {value}")


def _release_summary(release_dir, release, suites, results, dry_run, signature, catalogue):
    """Merge compatible subset results into the complete required-site matrix."""
    path = release_dir / "reports" / "release-summary.json"
    previous = (
        json.loads(path.read_text(encoding="utf-8")) if path.is_file() and not dry_run else {}
    )
    records = (
        {r["id"]: r for r in previous.get("results", [])}
        if previous.get("campaign_sha256") == signature
        else {}
    )
    _invalidate_dependents(records, results, catalogue)
    records.update({r["id"]: r for r in results})
    required = []
    for site, suite in suites.items():
        for test_id in suite["required_tests"]:
            key = f"{test_id}.{site}"
            records.setdefault(
                key,
                {
                    "id": key,
                    "test": test_id,
                    "site": site,
                    "report": None,
                    "status": "not_run",
                    "execution": "not_run",
                    "reason": "Not selected or not executed.",
                },
            )
            required.append(key)
    return {
        "release": release["release_label"],
        "baseline": release.get("baseline"),
        "required_sites": list(suites),
        "dry_run": dry_run,
        "campaign_sha256": signature,
        "qualified": all(records[key]["status"] == "pass" for key in required),
        "results": sorted(records.values(), key=lambda r: (r["site"], r["test"])),
    }


def _invalidate_dependents(records, results, catalogue):
    """Retain subset results only when their upstream evidence was not rerun."""
    for record in records.values():
        dependencies = set(_dependency_order([record["test"]], catalogue))
        for result in results:
            changed = result["site"] == record["site"] and result["id"] != record["id"]
            definition = catalogue[result["test"]]
            upstream = (
                result["test"] in dependencies
                or definition.get("produces_production")
                or definition.get("affects_production")
            )
            if changed and upstream:
                record.update(
                    status="not_run",
                    execution="not_run",
                    reason="Upstream evidence changed; rerun required.",
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
    payload = [release, catalogue, context, suites, products, rules, model_source]
    for definition in catalogue.values():
        payload.append(
            _resolve_workflow(template_dir, release_dir, definition["workflow"]).read_text(
                encoding="utf-8"
            )
        )
    runtime_file = template_dir / "run_time.yml"
    if runtime_file.is_file():
        payload.append(runtime_file.read_text(encoding="utf-8"))
    for profile in sorted((template_dir / "profiles").glob("*.yml")):
        payload.append(profile.read_text(encoding="utf-8"))
    for environment_file in sorted((template_dir / "profiles").glob("*.env")):
        payload.append(environment_file.read_text(encoding="utf-8"))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_acceptance(release_dir, template_dir):
    """Load the shared acceptance contracts and reject unsupported overrides."""
    products = _load_yaml(template_dir / "acceptance" / "expected-products.yml").get("tests")
    rules = _load_yaml(template_dir / "acceptance" / "thresholds.yml").get("rules")
    if not isinstance(products, dict) or not isinstance(rules, dict):
        raise ValueError("Expected products and acceptance rules must be mappings.")
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
    return {
        **catalogue[test_id],
        "products": products,
        "rule": rule,
        "expected_changes": release.get("expected_changes", []),
        "baseline": release.get("baseline"),
        "release_label": release.get("release_label"),
    }


def _validate_rule(rule, rule_id):
    """Fail malformed acceptance rules before any workflow can execute."""
    if not isinstance(rule, dict) or (rule_id and rule.get("mode") not in {"advisory", "blocking"}):
        raise ValueError(f"Invalid acceptance rule: {rule_id}")
    metrics = rule.get("metrics", [])
    if not isinstance(metrics, list):
        raise ValueError("Acceptance metrics must be a list.")
    for metric in metrics:
        _validate_metric_rule(metric)


def _validate_metric_rule(metric):
    """Validate a metric's JSON selector and finite numerical limits."""
    if not isinstance(metric, dict):
        raise ValueError("Each acceptance metric needs file and key.")
    if not isinstance(metric.get("file"), str) or not isinstance(metric.get("key"), str):
        raise ValueError("Each acceptance metric needs file and key.")
    for limit in ("minimum", "maximum"):
        if limit not in metric:
            continue
        value = metric[limit]
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"Invalid acceptance {limit}: {value}")


def run_release(
    release_dir,
    *,
    context_file=None,
    template_dir=None,
    sites=None,
    tests=None,
    dry_run=False,
    allow_production=False,
    overwrite=False,
    application_args=None,
):
    """Validate and run the selected site/test pairs for one release.

    The release directory contains a release definition and site files. Test
    implementations are read from the shared template catalogue.

    Parameters
    ----------
    release_dir : str or pathlib.Path
        Directory containing release.yml and required site suites.
    context_file : str or pathlib.Path or None
        Candidate, baseline, production-configuration paths and release label.
    template_dir : str or pathlib.Path or None
        Shared template; discovered above the release directory by default.
    sites, tests : iterable of str or None
        Selected sites/tests; comparison dependencies are included.
    dry_run : bool
        Validate the complete selection without writes or execution.
    allow_production : bool
        Permit explicitly selected production using a previously reviewed grid.
    overwrite : bool
        Archive and retry failed production only after its jobs have ended.
    application_args : dict or None
        Common runner arguments such as model-source and environment overrides.

    Returns
    -------
    dict
        Full required-site result matrix and qualification status.

    Raises
    ------
    ValueError
        Invalid definitions, context, selection, or production permission.
    RuntimeError
        A workflow failed or lacks required evidence.
    """
    release_dir = Path(release_dir).resolve()
    release = _load_yaml(release_dir / _RELEASE_FILE_NAME)
    template_dir = _resolve_template_dir(release_dir, template_dir)
    catalogue = _load_catalogue(release, release_dir, template_dir)
    context = _load_context(context_file)
    if not context:
        raise ValueError("A context_file is required for a science-test run.")
    context["__SCIENCE_RELEASE_LABEL__"] = release["release_label"]
    suites = _load_suites(release_dir, release, catalogue)
    requested_sites = _select(sites, suites, "site")
    requested_tests = _select(tests, catalogue, "test") if tests is not None else None
    _validate_selected_context(
        release, context, requested_sites, requested_tests, suites, catalogue
    )
    acceptance = _load_acceptance(release_dir, template_dir)
    application_args = _model_source_arguments(application_args, context, release)
    application_args["science_overwrite"] = overwrite
    model_source = {
        key: str(application_args[key])
        for key in _MODEL_SOURCE_KEYS
        if application_args and application_args.get(key) is not None
    }
    signature = _campaign_signature(
        release_dir,
        template_dir,
        release,
        catalogue,
        context,
        suites,
        acceptance["products"],
        acceptance["rules"],
        model_source,
    )
    prepared = []
    run_id = get_uuid()
    for site in requested_sites:
        site_config = suites[site]
        selected = site_config["required_tests"] if requested_tests is None else requested_tests
        for test_id in _dependency_order(selected, catalogue):
            _validate_definition(test_id, site, catalogue[test_id])
            definition = _test_definition(test_id, catalogue, acceptance, release)
            _check_production_selection(
                test_id, definition, requested_tests, allow_production, dry_run
            )
            args = (release_dir, template_dir, context, site, site_config, test_id, definition)
            prepared.append(
                (
                    args,
                    _run_test(
                        *args, dry_run=True, run_id=run_id, application_args=application_args
                    ),
                )
            )
    results = _execute_selection(prepared, dry_run, run_id, application_args)
    summary = _release_summary(release_dir, release, suites, results, dry_run, signature, catalogue)
    if not dry_run:
        _write_summary(release_dir / "reports", summary)
    failures = [
        result for result in results if result["status"] in {"fail", "incomplete", "blocked"}
    ]
    if failures:
        details = "; ".join(
            f"{result['id']}: {result.get('error') or result.get('reason') or result['status']}"
            for result in failures
        )
        raise RuntimeError(f"Science tests failed: {details}")
    return summary


def setup_release(release_dir, template_dir=None, context_file=None):
    """Create a new release directory from the shared science-test templates."""
    release_dir = Path(release_dir).resolve()
    template_dir = _resolve_template_dir(release_dir, template_dir)
    release_label = release_dir.parent.name
    if not _VALID_NAME.fullmatch(release_label):
        release_label = "candidate"
    site_templates = sorted(
        path for path in (template_dir / "sites").glob("*.yml") if path.is_file()
    )
    if not site_templates:
        raise FileNotFoundError(f"Science-test site templates not found: {template_dir / 'sites'}")
    template_paths = [
        _RELEASE_FILE_NAME,
        *(path.relative_to(template_dir) for path in site_templates),
    ]
    targets = [release_dir / path for path in (*template_paths, "context.yml")]
    existing = [path for path in targets if path.exists()]
    if existing:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Science-test setup files already exist: {paths}")
    context_source = Path(context_file) if context_file else template_dir / "context.example.yml"
    sources = [template_dir / path for path in template_paths] + [context_source]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Science-test setup templates not found: {paths}")
    for source, target in zip(sources, targets):
        target.parent.mkdir(parents=True, exist_ok=True)
        contents = source.read_text(encoding="utf-8").replace(_SETUP_RELEASE_LABEL, release_label)
        if not contents.startswith("---"):
            contents = "---\n" + contents
        target.write_text(contents, encoding="utf-8")
    logger.info("Created science-test setup in %s", release_dir)
    logger.info(
        "Edit these files before the dry run: %s",
        ", ".join(str(release_dir / path) for path in ("context.yml", _RELEASE_FILE_NAME)),
    )
    logger.info(
        "Review site settings in: %s",
        ", ".join(str(release_dir / path) for path in template_paths[1:]),
    )
    return release_dir


def _model_source_arguments(application_args, context, release):
    """Apply campaign model settings over runner defaults."""
    args = dict(application_args or {})
    for configuration in (context, release):
        args.update({key: configuration[key] for key in _MODEL_SOURCE_KEYS if key in configuration})
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


def _run_test(
    release_dir,
    template_dir,
    context,
    site,
    site_config,
    test_id,
    definition,
    *,
    dry_run,
    run_id=None,
    application_args=None,
):
    run_id = run_id or get_uuid()
    report_root = release_dir / "reports" / site / test_id
    work = Path(context["__SCIENCE_CANDIDATE_PATH__"]) / "work" / "science-tests" / site / test_id
    workflow = _resolve_workflow(template_dir, release_dir, definition["workflow"])
    replacements = _site_replacements(context, site, site_config, work)
    replacements["__CONFIG_DIRECTORY__"] = str(workflow.parent.resolve())
    resolved = replace_placeholders_recursively(
        _load_yaml(workflow),
        replacements,
    )
    if (
        not definition.get("produces_production")
        and not definition.get("collect_production")
        and "runtime_environment" not in resolved
    ):
        runtime = _shared_runtime_environment(template_dir, replacements)
        if runtime is not None:
            resolved["runtime_environment"] = runtime
    for application in resolved.get("applications", []):
        allowed = application.get("configuration", {}).get("compare_by", [])
        if set(allowed) - set(definition.get("expected_changes", [])):
            raise ValueError(f"Undeclared expected comparison differences: {allowed}")
    schema.validate_dict_using_schema(
        resolved,
        schema_file=SCHEMA_PATH / "application_workflow.metaschema.yml",
        offline=True,
    )
    args = {
        "ignore_runtime_environment": False,
        "ignore_existing_parameter_version": False,
        **(application_args or {}),
        "config_file": str(workflow),
        "steps": None,
        "log_file": str(work / "execution.log"),
        "provenance_path": str(work / "provenance" / "resolved-workflow.yml"),
    }
    args.setdefault("runtime_environment", resolved.get("runtime_environment"))
    configurations, runtime, *_ = prepare_workflow(args, replacements=replacements)

    result = {
        "id": f"{test_id}.{site}",
        "test": test_id,
        "site": site,
        "workflow": _relative_path(workflow, template_dir),
        "baseline": definition.get("baseline"),
        "candidate": definition.get("release_label"),
        "execution_provenance": work.relative_to(
            Path(context["__SCIENCE_CANDIDATE_PATH__"])
        ).as_posix(),
        "report": None,
        "run_id": run_id,
        "runtime": _runtime_description(runtime, args["ignore_runtime_environment"]),
        "output_paths": _test_output_paths(configurations, replacements),
        "execution": "planned" if dry_run else "completed",
        "status": "planned" if dry_run else "pass",
    }
    if dry_run:
        return result

    try:
        _prepare_test_retry(definition, replacements, args, run_id, work, report_root)
        _clear_test_directory(work)
        _check_inputs(definition, replacements)
        result.update(_execute_test_workflow(definition, args, replacements))
    except _AcceptanceError as exc:
        result["status"] = "fail"
        result["error"] = str(exc)
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
        JobExecutionError,
    ) as exc:
        result["status"] = "incomplete"
        result["execution"] = "failed"
        result["error"] = str(exc)
    _write_test_report(work, report_root, release_dir, result)
    logger.info(
        "%s: %s%s",
        result["id"],
        result["status"],
        f" - {result['reason']}" if result.get("reason") else "",
    )
    return result


def _test_output_paths(configurations, replacements):
    """Use application output paths or the candidate path for native collection."""
    paths = [str(config["configuration"]["output_path"]) for config in configurations]
    return list(dict.fromkeys(paths)) or [replacements["__SCIENCE_CANDIDATE_SITE_PATH__"]]


def _execute_test_workflow(definition, args, replacements):
    """Submit, collect, or execute a regular science workflow."""
    if definition.get("collect_production"):
        return _collect_production(definition, replacements)
    run_applications(args, replacements=replacements)
    validated = _evaluate_products(definition, replacements)
    if definition.get("produces_production"):
        return {
            "status": "submitted",
            "execution": "submitted",
            "reason": "Jobs submitted; run --test production.gamma.collect to collect results.",
        }
    return validated


def _collect_production(definition, replacements):
    """Check all production submissions once, validating only completed results."""
    candidate = Path(replacements["__SCIENCE_CANDIDATE_SITE_PATH__"])
    manifests = _matched_files(str(candidate / f"**/{_SUBMISSION_FILE_NAME}"))
    pending = False
    for manifest in manifests:
        submission = load_submission(manifest)
        if submission.backend != "htcondor":
            raise ValueError(f"Collection is not supported for backend {submission.backend}.")
        if submission.metadata.get("state") != "completed":
            pending = collect_submission(submission) is None or pending
    if pending:
        return {
            "status": "pending",
            "execution": "pending",
            "reason": "Production jobs are still queued; rerun collection after they finish.",
        }
    _check_inputs({"requires_completed_production": True}, replacements)
    return _evaluate_products(definition, replacements)


def _prepare_test_retry(definition, replacements, args, run_id, work, report):
    """Archive a failed production before an explicitly requested retry."""
    if not args.get("science_overwrite") or not definition.get("produces_production"):
        return
    _check_inputs({"requires": definition.get("requires", [])}, replacements)
    candidate = Path(replacements["__SCIENCE_CANDIDATE_SITE_PATH__"])
    submissions = list(candidate.rglob(_SUBMISSION_FILE_NAME))
    if not submissions:
        return
    for manifest in submissions:
        _validate_production_retry(manifest)
    for source, destination in (
        (candidate, candidate.parent / "archive" / run_id / candidate.name),
        (work, work.parents[1] / "archive" / run_id / work.parent.name / work.name),
        (report, report.parents[1] / "archive" / run_id / report.parent.name / report.name),
    ):
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            logger.info("Archived previous run: %s", destination)


def _validate_production_retry(manifest):
    """Require a failed manifest and scheduler confirmation that every job ended."""
    submission = load_submission(manifest)
    if submission.metadata.get("state") != "failed":
        raise ValueError(f"Cannot overwrite {manifest}: only failed production can be retried.")
    if submission.backend != "htcondor":
        raise ValueError(f"Cannot confirm finished jobs for backend {submission.backend}.")
    if not get_backend(submission.backend).is_finished(submission):
        raise ValueError(f"Cannot overwrite {manifest}: jobs are still in the HTCondor queue.")


def _clear_test_directory(path):
    """Remove artifacts from the previous execution of this named test."""
    if path.exists():
        shutil.rmtree(path)


def _write_test_report(work, report_root, release_dir, result):
    """Publish a small report even when execution or acceptance fails."""
    _clear_test_directory(report_root)
    source = work / "collected"
    if source.exists():
        shutil.copytree(source, report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    result["report"] = report_root.relative_to(release_dir).as_posix()
    (report_root / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    reason = (
        result.get("error")
        or result.get("reason")
        or "Required products and configured rules passed."
    )
    (report_root / "summary.md").write_text(
        f"# {result['id']}\n\nStatus: {result['status']}\n\n"
        f"Run ID: {result['run_id']}\n\n"
        f"Baseline: {result['baseline']}; candidate: {result['candidate']}\n\n{reason}\n",
        encoding="utf-8",
    )


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
            "__SCIENCE_REPORT_PATH__": str(work / "output"),
            "__SCIENCE_COLLECTION_PATH__": str(work / "collected"),
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
    """Load the optional science-test context mapping."""
    if context_file is None:
        return {}
    return _load_yaml(Path(context_file))


def _load_yaml(path):
    if not path.exists():
        raise FileNotFoundError(path)
    data = ascii_handler.collect_data_from_file(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def _write_summary(report_dir, summary):
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "release-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# Science-test summary: {summary['release']}",
        "",
        f"Qualified: {summary['qualified']}",
        "",
        "| Site | Test | Execution | Status | Report |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in summary["results"]:
        report = result.get("report")
        link = (
            f"[report]({Path(report).relative_to('reports').as_posix()}/summary.md)"
            if report
            else "-"
        )
        lines.append(
            f"| {result['site']} | {result['test']} | {result['execution']} | "
            f"{result['status']} | {link} |"
        )
    (report_dir / "release-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

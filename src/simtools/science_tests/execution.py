"""Prepare, execute, and collect science-test workflows."""

import json
import logging
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

import yaml
from astropy.io.registry import IORegistryError

from simtools.constants import SCHEMA_PATH
from simtools.data_model import schema
from simtools.io.ascii_handler import to_builtin
from simtools.job_execution.backends.registry import get_backend
from simtools.job_execution.execution import collect_submission, load_submission
from simtools.job_execution.job_manager import JobExecutionError
from simtools.production_configuration.compatibility import RECORD_NAME, _preview
from simtools.production_configuration.production_file_selection import (
    _values_match,
    discover_product_manifests,
    normalize_for_comparison,
    select_file_groups,
)
from simtools.production_configuration.trigger_histograms import load_trigger_histograms
from simtools.runners.simtools_runner import (
    _apply_execution_overrides,
    _get_explicit_env_file,
    prepare_workflow,
    run_applications,
)
from simtools.science_tests.definitions import (
    _EXECUTION_LOG_FILE_NAME,
    _SUBMISSION_FILE_NAME,
    _WORKFLOW_FILE_NAME,
    _AcceptanceError,
    _check_inputs,
    _check_reviewed_production,
    _dependency_order,
    _evaluate_products,
    _load_yaml,
    _matched_files,
    _relative_path,
    _resolve_workflow,
    _shared_runtime_environment,
    _site_replacements,
)
from simtools.science_tests.reports import (
    _clear_test_directory,
    _write_test_report,
)
from simtools.utils.general import get_uuid, replace_placeholders_recursively
from simtools.version import __version__

_PRODUCES_PRODUCTION = "produces_production"
_ENCODING = "utf-8"
_DEPENDENCY_COMPATIBILITY = "dependency_compatibility"
_CANDIDATE_SITE_PATH = "__SCIENCE_CANDIDATE_SITE_PATH__"

_CONFIGURATION = "configuration"
_APPLICATION = "application"

logger = logging.getLogger(__name__)


def _execute_selection(prepared, dry_run, run_id, application_args, catalogue):
    """Execute dependency order and record blocked downstream tests."""
    results = []
    for args, planned in prepared:
        if dry_run:
            _log_test_execution(planned, "Planned")
            results.append(planned)
            continue
        dependencies = _selected_ancestor_ids(planned, results, catalogue)
        if any(r["id"] in dependencies and r["status"] not in {"pass", "warn"} for r in results):
            status = _dependency_status(results, dependencies)
            results.append(
                {
                    **planned,
                    "status": status,
                    "execution": status,
                    "reason": "Dependency failed or is not yet complete.",
                }
            )
        else:
            _log_test_execution(planned, "Running")
            results.append(
                _run_test(*args, dry_run=False, run_id=run_id, application_args=application_args)
            )
    return results


def _selected_ancestor_ids(planned, results, catalogue):
    """Return every selected upstream test for a planned test and site."""
    ancestors = set(_dependency_order([planned["test"]], catalogue))
    ancestors.discard(planned["test"])
    selected = {result["test"] for result in results if result["site"] == planned["site"]}
    return {f"{test_id}.{planned['site']}" for test_id in ancestors & selected}


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
    if definition.get("model_version"):
        replacements["__SCIENCE_MODEL_VERSION__"] = str(definition["model_version"])
    replacements["__CONFIG_DIRECTORY__"] = str(workflow.parent.resolve())
    resolved = _resolved_test_workflow(template_dir, workflow, replacements, definition)
    args = {
        "ignore_runtime_environment": False,
        "ignore_existing_parameter_version": False,
        **(application_args or {}),
        "config_file": str(workflow),
        "steps": None,
        "log_file": str(work / _EXECUTION_LOG_FILE_NAME),
        "provenance_path": str(work / _WORKFLOW_FILE_NAME),
    }
    _apply_execution_overrides(
        resolved.get("applications", []),
        args,
        {
            key: args[key]
            for key in (
                "simulation_models_path",
                "simulation_models_git_path",
                "simulation_models_git_revision",
            )
            if args.get(key)
        },
        _get_explicit_env_file(args),
    )
    args["science_workflow"] = resolved
    args.setdefault("runtime_environment", resolved.get("runtime_environment"))
    configurations, runtime, *_ = prepare_workflow(args, replacements=replacements)

    result = {
        "id": f"{test_id}.{site}",
        "test": test_id,
        "site": site,
        "workflow": _relative_path(workflow, template_dir),
        "baseline": definition.get("baseline"),
        "candidate": definition.get("release_label"),
        "execution_provenance": str(work),
        "report": None,
        "run_id": run_id,
        "runtime": _runtime_description(runtime, args["ignore_runtime_environment"]),
        "software": {
            "simtools": __version__,
            "simulation_models": (application_args or {}).get("simulation_models_git_revision"),
        },
        "output_paths": _test_output_paths(configurations, replacements),
        "execution": "planned" if dry_run else "completed",
        "status": "planned" if dry_run else "pass",
    }
    if dry_run:
        if definition.get(_PRODUCES_PRODUCTION):
            result[_DEPENDENCY_COMPATIBILITY] = _preview_production(
                configurations, args, replacements
            )
        return result

    _execute_and_record_test(
        result, definition, replacements, args, run_id, work, report_root, resolved
    )
    _write_test_report(work, report_root, release_dir, result)
    logger.info(
        "%s: %s%s",
        result["id"],
        result["status"],
        f" - {result['reason']}" if result.get("reason") else "",
    )
    return result


def _preview_production(configurations, args, replacements):
    """Report the effective workflow selection, including explicit per-step overrides."""
    selected = dict(args)
    image = replacements.get("__SCIENCE_CONTAINER_IMAGE_PATH__")
    for application in configurations:
        if application.get(_APPLICATION) == "simtools-simulate-prod":
            selected.update(application.get(_CONFIGURATION, {}))
            backend = selected.get("backend_config")
            if isinstance(backend, dict):
                image = backend.get("container_image")
            break
    return _preview(selected, image)


def _resolved_test_workflow(template_dir, workflow, replacements, definition):
    """Resolve, customize, and validate one workflow before it is prepared."""
    resolved = replace_placeholders_recursively(_load_yaml(workflow), replacements)
    _add_shared_runtime(resolved, template_dir, replacements, definition)
    _configure_comparison_applications(resolved, definition)
    _disable_collection_inventory(resolved)
    schema.validate_dict_using_schema(
        resolved, schema_file=SCHEMA_PATH / "application_workflow.metaschema.yml", offline=True
    )
    return resolved


def _disable_collection_inventory(resolved):
    """Avoid a redundant checksum inventory for in-place science-test collection."""
    collection = resolved.get("collection")
    if isinstance(collection, dict):
        collection["write_inventory"] = False
    elif isinstance(collection, list):
        for entry in collection:
            if isinstance(entry, dict):
                entry["write_inventory"] = False


def _add_shared_runtime(resolved, template_dir, replacements, definition):
    """Apply the shared runtime to non-production workflows without one."""
    needs_runtime = not definition.get(_PRODUCES_PRODUCTION) and not definition.get(
        "collect_production"
    )
    if needs_runtime and "runtime_environment" not in resolved:
        runtime = _shared_runtime_environment(template_dir, replacements)
        if runtime is not None:
            resolved["runtime_environment"] = runtime


def _configure_comparison_applications(resolved, definition):
    """Apply release comparison settings and reject undeclared differences."""
    for application in resolved.get("applications", []):
        configuration = application.get(_CONFIGURATION, {})
        if application.get(_APPLICATION) == "simtools-compare-productions":
            configuration["report_mode"] = definition.get("report_mode", "compact")
            configuration.update(definition.get("comparison_parameters", {}))
            _set_event_comparison_labels(configuration, definition)
        allowed = configuration.get("compare_by", [])
        if set(allowed) - set(definition.get("expected_changes", [])):
            raise ValueError(f"Undeclared expected comparison differences: {allowed}")


def _set_event_comparison_labels(configuration, definition):
    """Use release identities in event-comparison legends."""
    if configuration.get("comparison_level") in {"events", "computing"}:
        configuration["baseline_label"] = definition["baseline"]
        configuration["candidate_label"] = definition["release_label"]


def _execute_and_record_test(
    result, definition, replacements, args, run_id, work, report_root, resolved
):
    """Execute one test while retaining a useful failure result."""
    try:
        _prepare_test_retry(definition, replacements, args, run_id, work, report_root)
        _check_inputs(definition, replacements)
        if definition.get("acceptance_rule") == "gamma-events":
            for root_key in (
                _CANDIDATE_SITE_PATH,
                "__SCIENCE_BASELINE_SITE_PATH__",
            ):
                _validate_derived_products(
                    {
                        "production_path": replacements[root_key],
                        "output_path": str(
                            Path(replacements[root_key]) / "derived/trigger_histograms"
                        ),
                    }
                )
        _run_or_reuse_test(result, definition, replacements, args, work, resolved)
    except _AcceptanceError as exc:
        result.update(status="fail", error=str(exc))
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
        JobExecutionError,
    ) as exc:
        result.update(status="incomplete", execution="failed", error=str(exc))


def _run_or_reuse_test(result, definition, replacements, args, work, resolved):
    """Reuse complete derivations or execute the workflow after preparing its work area."""
    if _reuse_derived_product(definition, replacements, args):
        result.update(_evaluate_products(definition, replacements))
        result.update(
            execution="reused",
            reason="Reused complete derived products; use --overwrite to rebuild them.",
        )
        return
    _prepare_derived_rebuild(definition, replacements, args)
    _clear_test_directory(work)
    args["config_file"] = str(_materialize_effective_workflow(work, resolved))
    try:
        result.update(_execute_test_workflow(definition, args, replacements))
    finally:
        log_path = work / _EXECUTION_LOG_FILE_NAME
        if log_path.is_file():
            result["execution_log"] = str(log_path)
    if definition.get("derived_output"):
        for application in resolved["applications"]:
            _validate_derived_products(application[_CONFIGURATION])


def _reuse_derived_product(definition, replacements, args):
    """Reuse only readable, current products covering every selected input group."""
    if not definition.get("derived_output") or args.get("science_overwrite"):
        return False
    output = Path(replace_placeholders_recursively(definition["derived_output"], replacements))
    if not output.exists() or not any(output.iterdir()):
        return False
    workflow = args["science_workflow"]
    previous = Path(args["provenance_path"])
    try:
        if not previous.is_file():
            raise ValueError("No recorded derivation settings")
        old = _load_yaml(previous)
        current_configs = [app[_CONFIGURATION] for app in workflow["applications"]]
        old_configs = [app[_CONFIGURATION] for app in old["applications"]]
        if _reuse_settings(current_configs) != _reuse_settings(old_configs):
            raise ValueError("Derivation settings changed")
        for configuration in current_configs:
            _validate_derived_products(configuration)
        _evaluate_products(definition, replacements)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(
            f"Derived products are incomplete or stale: {exc}. "
            "Rerun derive.trigger_histograms with --overwrite to rebuild."
        ) from exc
    return True


def _reuse_settings(configurations):
    """Compare effective settings while ignoring per-execution bookkeeping."""
    ignored = {"activity_id", "disable_log_file", "label"}
    return normalize_for_comparison(
        [
            {key: value for key, value in config.items() if key not in ignored}
            for config in configurations
        ]
    )


def _validate_derived_products(configuration):
    """Check histogram metadata, run coverage, and source freshness without publishing it."""
    groups = select_file_groups(
        configuration["production_path"],
        selections=configuration.get("select"),
        require_complete_runs=True,
    )["groups"]
    products = discover_product_manifests(configuration["output_path"], "trigger_histograms")
    by_key = {}
    for product in products:
        key = normalize_for_comparison(product.data[_CONFIGURATION])
        if key in by_key:
            raise ValueError("Duplicate derived configuration")
        by_key[key] = product
    if set(by_key) != {normalize_for_comparison(group.configuration) for group in groups}:
        raise ValueError("Derived configuration coverage differs from production")
    for group in groups:
        product = by_key[normalize_for_comparison(group.configuration)]
        _validate_derived_group(group, product, configuration)


def _validate_derived_group(group, product, configuration):
    """Require a readable histogram built from the current input runs and settings."""
    data = product.data
    if sorted(data["input_run_numbers"]) != sorted(group.run_numbers):
        raise ValueError("Derived run coverage differs from production")
    sources = [
        (product.directory / path).resolve() for path in data["input_files"]["reduced_event_data"]
    ]
    if set(sources) != {path.resolve() for path in group.file_paths}:
        raise ValueError("Derived input files differ from production")
    output_files = data["files"]["trigger_histograms"]
    if len(output_files) != 1:
        raise ValueError("Expected one histogram per configuration")
    output = product.directory / output_files[0]
    try:
        metadata, bins = load_trigger_histograms(output)
    except (OSError, ValueError, IORegistryError) as exc:
        raise ValueError(f"Unreadable trigger histogram: {output}") from exc
    if not metadata or not bins:
        raise ValueError("Empty trigger histogram")
    if any(path.stat().st_mtime_ns > output.stat().st_mtime_ns for path in group.file_paths):
        raise ValueError("Production inputs changed after derivation")
    for key, value in data["histogram_settings"].items():
        if key in configuration and not _values_match(value, str(configuration[key])):
            raise ValueError(f"Histogram setting changed: {key}")


def _prepare_derived_rebuild(definition, replacements, args):
    """Remove only this test's derived output when an explicit rebuild was requested."""
    derived_output = definition.get("derived_output")
    if not args.get("science_overwrite") or not derived_output:
        return
    path = Path(replace_placeholders_recursively(derived_output, replacements))
    if path.exists():
        shutil.rmtree(path)
        logger.info("Removed derived output for explicit rebuild: %s", path)


def _materialize_effective_workflow(work, resolved):
    """Write the effective workflow, later replaced by its resolved execution record."""
    work.mkdir(parents=True, exist_ok=True)
    path = work / _WORKFLOW_FILE_NAME
    path.write_text(
        yaml.safe_dump(to_builtin(resolved), sort_keys=False, explicit_start=True),
        encoding=_ENCODING,
    )
    return path


def _test_output_paths(configurations, replacements):
    """Use application output paths or the candidate path for native collection."""
    paths = [str(config[_CONFIGURATION]["output_path"]) for config in configurations]
    return list(dict.fromkeys(paths)) or [replacements[_CANDIDATE_SITE_PATH]]


def _execute_test_workflow(definition, args, replacements):
    """Submit, collect, or execute a regular science workflow."""
    if definition.get("collect_production"):
        return _collect_production(definition, replacements)
    with _selected_model_source_environment(args):
        run_applications(args, replacements=replacements)
    validated = _evaluate_products(definition, replacements)
    if definition.get(_PRODUCES_PRODUCTION):
        record = Path(replacements[_CANDIDATE_SITE_PATH]) / RECORD_NAME
        compatibility = (
            json.loads(record.read_text(encoding=_ENCODING)) if record.is_file() else None
        )
        return {
            _DEPENDENCY_COMPATIBILITY: compatibility,
            "status": "submitted",
            "execution": "submitted",
            "reason": "Jobs submitted; run --test production.gamma.collect to collect results.",
        }
    return validated


@contextmanager
def _selected_model_source_environment(args):
    """Keep an inactive model source from a parent .env out of child applications."""
    if args.get("simulation_models_git_path"):
        inactive = "SIMTOOLS_SIMULATION_MODELS_PATH"
    elif args.get("simulation_models_path"):
        inactive = "SIMTOOLS_SIMULATION_MODELS_GIT_PATH"
    else:
        yield
        return
    previous = os.environ.get(inactive)
    os.environ[inactive] = ""
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(inactive, None)
        else:
            os.environ[inactive] = previous


def _collect_production(definition, replacements):
    """Check all production submissions once, validating only completed results."""
    candidate = Path(replacements[_CANDIDATE_SITE_PATH])
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
    coverage = (
        _check_reviewed_production(definition, replacements)
        if definition.get("reviewed_grid")
        else {}
    )
    return _evaluate_products(definition, replacements) | {"coverage": coverage}


def _prepare_test_retry(definition, replacements, args, run_id, work, report):
    """Archive a failed production before an explicitly requested retry."""
    if not args.get("science_overwrite") or not definition.get(_PRODUCES_PRODUCTION):
        return
    _check_inputs({"requires": definition.get("requires", [])}, replacements)
    candidate = Path(replacements[_CANDIDATE_SITE_PATH])
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

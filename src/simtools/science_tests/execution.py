"""Small runner for reusable, site-parameterised release science tests."""

import logging
import shutil
import subprocess
from pathlib import Path

import yaml

from simtools.constants import SCHEMA_PATH
from simtools.data_model import schema
from simtools.job_execution.backends.registry import get_backend
from simtools.job_execution.execution import collect_submission, load_submission
from simtools.job_execution.job_manager import JobExecutionError
from simtools.runners.simtools_runner import prepare_workflow, run_applications
from simtools.science_tests.definitions import (
    _EXECUTION_LOG_FILE_NAME,
    _SUBMISSION_FILE_NAME,
    _WORKFLOW_FILE_NAME,
    _AcceptanceError,
    _check_inputs,
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
        "software": {
            "simtools": __version__,
            "simulation_models": (application_args or {}).get("simulation_models_git_revision"),
        },
        "output_paths": _test_output_paths(configurations, replacements),
        "execution": "planned" if dry_run else "completed",
        "status": "planned" if dry_run else "pass",
    }
    if dry_run:
        return result

    _execute_and_record_test(
        result, definition, replacements, args, run_id, work, report_root, resolved
    )
    result["execution_log"] = _relative_path(
        work / _EXECUTION_LOG_FILE_NAME, Path(context["__SCIENCE_CANDIDATE_PATH__"])
    )
    _write_test_report(work, report_root, release_dir, result)
    logger.info(
        "%s: %s%s",
        result["id"],
        result["status"],
        f" - {result['reason']}" if result.get("reason") else "",
    )
    return result


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
    needs_runtime = not definition.get("produces_production") and not definition.get(
        "collect_production"
    )
    if needs_runtime and "runtime_environment" not in resolved:
        runtime = _shared_runtime_environment(template_dir, replacements)
        if runtime is not None:
            resolved["runtime_environment"] = runtime


def _configure_comparison_applications(resolved, definition):
    """Apply release comparison settings and reject undeclared differences."""
    for application in resolved.get("applications", []):
        configuration = application.get("configuration", {})
        if application.get("application") == "simtools-compare-productions":
            configuration["report_mode"] = definition.get("report_mode", "compact")
            configuration.update(definition.get("comparison_parameters", {}))
            _set_event_comparison_labels(configuration, definition)
        allowed = configuration.get("compare_by", [])
        if set(allowed) - set(definition.get("expected_changes", [])):
            raise ValueError(f"Undeclared expected comparison differences: {allowed}")


def _set_event_comparison_labels(configuration, definition):
    """Use release identities in event-comparison legends."""
    if configuration.get("comparison_level") == "events":
        configuration.setdefault("baseline_label", definition["baseline"])
        configuration.setdefault("candidate_label", definition["release_label"])


def _execute_and_record_test(
    result, definition, replacements, args, run_id, work, report_root, resolved
):
    """Execute one test while retaining a useful failure result."""
    try:
        _prepare_test_retry(definition, replacements, args, run_id, work, report_root)
        _check_inputs(definition, replacements)
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
    result.update(_execute_test_workflow(definition, args, replacements))


def _reuse_derived_product(definition, replacements, args):
    """Reuse an existing complete derived product unless an explicit rebuild was requested."""
    if not definition.get("derived_output") or args.get("science_overwrite"):
        return False
    try:
        _evaluate_products(definition, replacements)
    except ValueError:
        return False
    return True


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
    path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    return path


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

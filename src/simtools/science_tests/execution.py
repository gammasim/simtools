"""Prepare, execute, and collect science-test workflows."""

import logging
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

import yaml

from simtools import dependencies
from simtools.constants import SCHEMA_PATH
from simtools.data_model import schema
from simtools.io.ascii_handler import to_builtin
from simtools.job_execution.backends.registry import get_backend
from simtools.job_execution.execution import collect_submission, load_submission
from simtools.job_execution.job_manager import JobExecutionError
from simtools.production_configuration.compatibility import _preview
from simtools.production_configuration.production_comparison import _validate_comparison_inputs
from simtools.production_configuration.production_file_selection import (
    normalize_for_comparison,
)
from simtools.production_configuration.trigger_histograms import _validate_derived_workflow
from simtools.runners.simtools_runner import (
    _apply_execution_overrides,
    _get_explicit_env_file,
    prepare_workflow,
    read_runtime_environment,
    run_applications,
)
from simtools.science_tests.camera_efficiency_results import (
    collect_legacy_results,
    sha256_file,
    write_json,
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
from simtools.utils import names
from simtools.utils.general import get_uuid, replace_placeholders_recursively
from simtools.version import __version__

_STATUS_KEY = "status"
_INCOMPLETE_STATUS = "incomplete"
_SCHEMA_VERSION_KEY = "schema_version"
_DIAGNOSTICS_KEY = "diagnostics"
_TELESCOPE_KEY = "telescope"

_MODEL_VERSION_KEY = "model_version"
_IGNORE_RUNTIME_KEY = "ignore_runtime_environment"
_SCIENCE_WORKFLOW_KEY = "science_workflow"
_RUNTIME_ENVIRONMENT_KEY = "runtime_environment"
_BASELINE_KEY = "baseline"
_RELEASE_LABEL_KEY = "release_label"
_GIT_MODELS_PATH_KEY = "simulation_models_git_path"
_MODELS_PATH_KEY = "simulation_models_path"
_MODELS_REVISION_KEY = "simulation_models_git_revision"
_WRITE_INVENTORY_KEY = "write_inventory"
_OUTPUT_PATH_KEY = "output_path"

_SCIENCE_CANDIDATE_SITE_PATH = "__SCIENCE_CANDIDATE_SITE_PATH__"
_APPLICATIONS_KEY = "applications"
_DERIVED_OUTPUT = "derived_output"
_SCIENCE_OVERWRITE = "science_overwrite"

_PRODUCES_PRODUCTION = "produces_production"
_APPLICATION = "application"
_CONFIGURATION = "configuration"
_CAMERA_TELESCOPES_KEY = "camera_telescopes"
_CAMERA_RESULT_TYPE = "telescope_throughput"
_LEGACY_CAMERA_RESULT_TYPE = "camera_efficiency"
_CAMERA_RESULT_TYPES = {_CAMERA_RESULT_TYPE, _LEGACY_CAMERA_RESULT_TYPE}
_CAMERA_MANIFEST_FILE = "telescope_throughput_manifest.json"
_LEGACY_CAMERA_VALIDATOR_APPLICATION = "simtools-validate-camera-efficiency"
_CAMERA_VALIDATOR_APPLICATION = "simtools-validate-telescope-throughput"
_CAMERA_VALIDATOR_APPLICATIONS = {
    _LEGACY_CAMERA_VALIDATOR_APPLICATION,
    _CAMERA_VALIDATOR_APPLICATION,
}
_SCIENCE_COMPARISON_APPLICATION = "simtools-compare-science-results"
_SIMTOOLS_KEY = "simtools"

logger = logging.getLogger(__name__)


def _execute_selection(prepared, dry_run, run_id, application_args, catalogue):
    """Execute dependency order and record blocked downstream tests."""
    results = []
    for args, planned in prepared:
        if dry_run:
            _log_test_execution(planned, "Planned")
            results.append(planned)
            continue
        dependency_ids = _selected_ancestor_ids(planned, results, catalogue)
        if any(
            r["id"] in dependency_ids and r[_STATUS_KEY] not in {"pass", "warn"} for r in results
        ):
            status = _dependency_status(results, dependency_ids)
            results.append(
                {
                    **planned,
                    _STATUS_KEY: status,
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


def _dependency_status(results, dependency_ids):
    """Distinguish queued dependencies from failed dependencies."""
    if all(
        r[_STATUS_KEY] in {"pass", "warn", "submitted", "pending"}
        for r in results
        if r["id"] in dependency_ids
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
    if definition.get(_MODEL_VERSION_KEY):
        replacements["__SCIENCE_MODEL_VERSION__"] = str(definition[_MODEL_VERSION_KEY])
    replacements["__CONFIG_DIRECTORY__"] = str(workflow.parent.resolve())
    resolved = _resolved_test_workflow(
        template_dir, workflow, replacements, definition, site_config
    )
    args = {
        _IGNORE_RUNTIME_KEY: False,
        "ignore_existing_parameter_version": False,
        **(application_args or {}),
        "config_file": str(workflow),
        "steps": None,
        "log_file": str(work / _EXECUTION_LOG_FILE_NAME),
        "provenance_path": str(work / _WORKFLOW_FILE_NAME),
    }
    _apply_execution_overrides(
        resolved.get(_APPLICATIONS_KEY, []),
        args,
        {
            key: args[key]
            for key in (
                _MODELS_PATH_KEY,
                _GIT_MODELS_PATH_KEY,
                _MODELS_REVISION_KEY,
            )
            if args.get(key)
        },
        _get_explicit_env_file(args),
    )
    args[_SCIENCE_WORKFLOW_KEY] = resolved
    args.setdefault(_RUNTIME_ENVIRONMENT_KEY, resolved.get(_RUNTIME_ENVIRONMENT_KEY))
    configurations, runtime, *_ = prepare_workflow(args, replacements=replacements)

    result = {
        "id": f"{test_id}.{site}",
        "test": test_id,
        "site": site,
        "workflow": _relative_path(workflow, template_dir),
        _BASELINE_KEY: definition.get(_BASELINE_KEY),
        "candidate": definition.get(_RELEASE_LABEL_KEY),
        "execution_provenance": str(work),
        "report": None,
        "run_id": run_id,
        "runtime": _runtime_description(runtime, args[_IGNORE_RUNTIME_KEY]),
        "software": {
            _SIMTOOLS_KEY: __version__,
            "simulation_models": (application_args or {}).get(_MODELS_REVISION_KEY),
        },
        "output_paths": _test_output_paths(configurations, replacements),
        "execution": "planned" if dry_run else "completed",
        _STATUS_KEY: "planned" if dry_run else "pass",
    }
    if definition.get(_PRODUCES_PRODUCTION):
        result["dependency_compatibility"] = _preview_production(
            configurations, args, replacements, runtime
        )
    if dry_run:
        return result

    _execute_and_record_test(
        result, definition, replacements, args, run_id, work, report_root, resolved
    )
    _write_test_report(
        work,
        report_root,
        release_dir,
        result,
        publish=definition.get("publish_report", True),
    )
    logger.info(
        "%s: %s%s",
        result["id"],
        result[_STATUS_KEY],
        f" - {result['reason']}" if result.get("reason") else "",
    )
    return result


def _preview_production(configurations, args, replacements, runtime):
    """Check the effective production configuration, including workflow overrides."""
    selected = dict(args)
    image = (
        runtime.get("image") if runtime else replacements.get("__SCIENCE_CONTAINER_IMAGE_PATH__")
    )
    for application in configurations:
        if application.get(_APPLICATION) == "simtools-simulate-prod":
            selected.update(application.get(_CONFIGURATION, {}))
            backend = selected.get("backend_config")
            if isinstance(backend, dict) and backend.get("container_image"):
                image = backend["container_image"]
            break
    return _preview(selected, image)


def _resolved_test_workflow(template_dir, workflow, replacements, definition, site_config=None):
    """Resolve, customize, and validate one workflow before it is prepared."""
    resolved = replace_placeholders_recursively(_load_yaml(workflow), replacements)
    _expand_camera_efficiency_telescopes(resolved, site_config or {}, replacements)
    _add_shared_runtime(resolved, template_dir, replacements, definition)
    _configure_comparison_applications(resolved, definition)
    _disable_collection_inventory(resolved)
    schema.validate_dict_using_schema(
        resolved, schema_file=SCHEMA_PATH / "application_workflow.metaschema.yml", offline=True
    )
    return resolved


def _expand_camera_efficiency_telescopes(resolved, site_config, replacements):
    """Create one isolated validator application for each configured telescope."""
    applications = resolved.get(_APPLICATIONS_KEY, [])
    expanded = []
    for application in applications:
        configuration = application.get(_CONFIGURATION, {})
        if application.get(_APPLICATION) == _SCIENCE_COMPARISON_APPLICATION:
            telescopes = site_config.get(_CAMERA_TELESCOPES_KEY)
            if telescopes:
                configuration["telescopes"] = telescopes
        if application.get(_APPLICATION) not in _CAMERA_VALIDATOR_APPLICATIONS:
            expanded.append(application)
            continue
        telescopes = _camera_telescopes(site_config, replacements)
        base_output = Path(configuration[_OUTPUT_PATH_KEY])
        for telescope in telescopes:
            selected = {**application, _CONFIGURATION: dict(configuration)}
            selected_config = selected[_CONFIGURATION]
            selected_config[_TELESCOPE_KEY] = telescope
            if len(telescopes) > 1:
                selected_config[_OUTPUT_PATH_KEY] = str(base_output / telescope)
            expanded.append(selected)
    resolved[_APPLICATIONS_KEY] = expanded


def _camera_telescopes(site_config, replacements):
    """Normalize the configured telescope list and reject duplicate output targets."""
    telescopes = site_config.get(_CAMERA_TELESCOPES_KEY)
    if telescopes is None:
        telescope = replacements.get("__SCIENCE_CAMERA_TELESCOPE__")
        telescopes = [telescope] if telescope else []
    if (
        not isinstance(telescopes, list)
        or not telescopes
        or any(not isinstance(telescope, str) or not telescope for telescope in telescopes)
    ):
        raise ValueError(
            f"Camera-efficiency tests require a non-empty {_CAMERA_TELESCOPES_KEY} list."
        )
    telescopes = [names.validate_array_element_name(telescope) for telescope in telescopes]
    if len(telescopes) != len(set(telescopes)):
        raise ValueError("Camera-efficiency telescope names must be unique after normalization.")
    return telescopes


def _disable_collection_inventory(resolved):
    """Avoid a redundant checksum inventory for in-place science-test collection."""
    collection = resolved.get("collection")
    if isinstance(collection, dict):
        collection[_WRITE_INVENTORY_KEY] = False
    elif isinstance(collection, list):
        for entry in collection:
            if isinstance(entry, dict):
                entry[_WRITE_INVENTORY_KEY] = False


def _add_shared_runtime(resolved, template_dir, replacements, definition):
    """Apply the shared runtime to non-production workflows without one."""
    needs_runtime = not definition.get(_PRODUCES_PRODUCTION) and not definition.get(
        "collect_production"
    )
    if needs_runtime and _RUNTIME_ENVIRONMENT_KEY not in resolved:
        runtime = _shared_runtime_environment(
            template_dir, replacements, definition.get("runtime_role")
        )
        if runtime is not None:
            resolved[_RUNTIME_ENVIRONMENT_KEY] = runtime


def _configure_comparison_applications(resolved, definition):
    """Apply release comparison settings and reject undeclared differences."""
    for application in resolved.get(_APPLICATIONS_KEY, []):
        configuration = application.get(_CONFIGURATION, {})
        if application.get(_APPLICATION) == "simtools-compare-productions":
            configuration["report_mode"] = definition.get("report_mode", "compact")
            configuration.update(definition.get("comparison_parameters", {}))
            _set_event_comparison_labels(configuration, definition)
        if application.get(_APPLICATION) == _SCIENCE_COMPARISON_APPLICATION:
            _set_comparison_labels(configuration, definition)
        allowed = configuration.get("compare_by", [])
        if set(allowed) - set(definition.get("expected_changes", [])):
            raise ValueError(f"Undeclared expected comparison differences: {allowed}")


def _set_event_comparison_labels(configuration, definition):
    """Use release identities in event-comparison legends."""
    if configuration.get("comparison_level") in {"events", "computing"}:
        _set_comparison_labels(configuration, definition)


def _set_comparison_labels(configuration, definition):
    """Use the selected release labels consistently in every comparison."""
    configuration["baseline_label"] = definition[_BASELINE_KEY]
    configuration["candidate_label"] = definition[_RELEASE_LABEL_KEY]


def _execute_and_record_test(
    result, definition, replacements, args, run_id, work, report_root, resolved
):
    """Execute one test while retaining a useful failure result."""
    try:
        _prepare_test_retry(definition, replacements, args, run_id, work, report_root)
        _check_inputs(definition, replacements)
        _validate_comparison_inputs(
            resolved.get(_APPLICATIONS_KEY, []),
            [
                replacements[key]
                for key in (_SCIENCE_CANDIDATE_SITE_PATH, "__SCIENCE_BASELINE_SITE_PATH__")
                if key in replacements
            ],
        )
        _run_or_reuse_test(result, definition, replacements, args, work, resolved)
    except _AcceptanceError as exc:
        result.update(status="warn", reason=f"Acceptance warning: {exc}")
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
        JobExecutionError,
    ) as exc:
        result.update(status=_INCOMPLETE_STATUS, execution="failed", error=str(exc))


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
        if definition.get(_DERIVED_OUTPUT) and result.get(_STATUS_KEY) in {"pass", "warn"}:
            _validate_derived_workflow(resolved[_APPLICATIONS_KEY])
    finally:
        log_path = work / _EXECUTION_LOG_FILE_NAME
        if log_path.is_file():
            result["execution_log"] = str(log_path)


def _reuse_derived_product(definition, replacements, args):
    """Reuse only readable, current products covering every selected input group."""
    if not definition.get(_DERIVED_OUTPUT) or args.get(_SCIENCE_OVERWRITE):
        return False
    output = Path(replace_placeholders_recursively(definition[_DERIVED_OUTPUT], replacements))
    if not output.exists() or not any(output.iterdir()):
        return False
    workflow = args[_SCIENCE_WORKFLOW_KEY]
    previous = Path(args["provenance_path"])
    try:
        if not previous.is_file():
            raise ValueError("No recorded derivation settings")
        old = _load_yaml(previous)
        current_configs = [app[_CONFIGURATION] for app in workflow[_APPLICATIONS_KEY]]
        old_configs = [app[_CONFIGURATION] for app in old[_APPLICATIONS_KEY]]
        if _reuse_settings(current_configs) != _reuse_settings(old_configs):
            raise ValueError("Derivation settings changed")
        _validate_derived_workflow(workflow[_APPLICATIONS_KEY])
        _evaluate_products(definition, replacements)
    except _AcceptanceError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(
            f"Derived products are incomplete or stale: {exc}. "
            "Rerun this derivation with --overwrite to rebuild."
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


def _prepare_derived_rebuild(definition, replacements, args):
    """Remove only this test's derived output when an explicit rebuild was requested."""
    derived_output = definition.get(_DERIVED_OUTPUT)
    if not args.get(_SCIENCE_OVERWRITE) or not derived_output:
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
        yaml.safe_dump(to_builtin(resolved), sort_keys=False, explicit_start=True), encoding="utf-8"
    )
    return path


def _test_output_paths(configurations, replacements):
    """Use application output paths or the candidate path for native collection."""
    paths = [str(config[_CONFIGURATION][_OUTPUT_PATH_KEY]) for config in configurations]
    return list(dict.fromkeys(paths)) or [replacements[_SCIENCE_CANDIDATE_SITE_PATH]]


def _execute_test_workflow(definition, args, replacements):
    """Submit, collect, or execute a regular science workflow."""
    if definition.get("collect_production"):
        return _collect_production(definition, replacements)
    camera_efficiency = definition.get("result_type") in _CAMERA_RESULT_TYPES
    if camera_efficiency:
        _invalidate_camera_efficiency_results(args)
    with _selected_model_source_environment(args):
        run_applications(args, replacements=replacements)
    if camera_efficiency:
        _collect_camera_efficiency_results(args)
    validated = _evaluate_products(definition, replacements)
    if definition.get(_PRODUCES_PRODUCTION):
        return {
            _STATUS_KEY: "submitted",
            "execution": "submitted",
            "reason": "Jobs submitted; run --test production.gamma.collect to collect results.",
        }
    return validated


def _camera_validation_configurations(args):
    """Return the effective configurations of this workflow's camera validators."""
    return [
        item.get(_CONFIGURATION, {})
        for item in args[_SCIENCE_WORKFLOW_KEY].get(_APPLICATIONS_KEY, [])
        if item.get(_APPLICATION) in _CAMERA_VALIDATOR_APPLICATIONS
    ]


def _invalidate_camera_efficiency_results(args):
    """Mark previous evidence incomplete before any validator can overwrite its tables."""
    for configuration in _camera_validation_configurations(args):
        write_json(
            Path(configuration[_OUTPUT_PATH_KEY]) / _CAMERA_MANIFEST_FILE,
            {
                _SCHEMA_VERSION_KEY: 1,
                _STATUS_KEY: _INCOMPLETE_STATUS,
                "case": {_TELESCOPE_KEY: configuration.get(_TELESCOPE_KEY)},
                _DIAGNOSTICS_KEY: ["Camera-efficiency execution has not completed successfully."],
            },
        )


def _collect_camera_efficiency_results(args):
    """Normalize camera outputs and record the software and image that ran them."""
    validations = _camera_validation_configurations(args)
    if not validations:
        raise ValueError("Camera-efficiency result type requires its validation application.")
    runtime = None if args.get(_IGNORE_RUNTIME_KEY) else args.get(_RUNTIME_ENVIRONMENT_KEY)
    image = runtime.get("image") if runtime else None
    image_identity = _local_image_identity(image)
    simtools_version = _runtime_simtools_version(runtime)
    for configuration in validations:
        output = Path(configuration[_OUTPUT_PATH_KEY])
        manifest = collect_legacy_results(output, configuration, image_identity, simtools_version)
        manifest_path = output / _CAMERA_MANIFEST_FILE
        write_json(manifest_path, manifest)


def _local_image_identity(image):
    """Hash a local runtime image when its file is available to the host."""
    if not image or not Path(image).is_file():
        return None
    return f"sha256:{sha256_file(image)}"


def _runtime_simtools_version(runtime):
    """Read the installed simtools version from the selected runtime manifest."""
    runtime_command = read_runtime_environment(runtime) if runtime else None
    if runtime_command is None:
        return __version__
    try:
        manifest = dependencies.get_dependency_manifest(runtime_command)
        software = manifest.get(_SIMTOOLS_KEY, {})
        version = software.get("version") if isinstance(software, dict) else software
        if version:
            return version
    except OSError, ValueError, subprocess.SubprocessError:
        pass
    result = subprocess.run(
        [
            *runtime_command,
            "python",
            "-c",
            "from simtools.version import __version__; print(__version__)",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    logger.warning("Could not read simtools version from the selected runtime.")
    return None


@contextmanager
def _selected_model_source_environment(args):
    """Keep an inactive model source from a parent .env out of child applications."""
    if args.get(_GIT_MODELS_PATH_KEY):
        inactive = "SIMTOOLS_SIMULATION_MODELS_PATH"
    elif args.get(_MODELS_PATH_KEY):
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
    candidate = Path(replacements[_SCIENCE_CANDIDATE_SITE_PATH])
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
            _STATUS_KEY: "pending",
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
    if not args.get(_SCIENCE_OVERWRITE) or not definition.get(_PRODUCES_PRODUCTION):
        return
    _check_inputs({"requires": definition.get("requires", [])}, replacements)
    candidate = Path(replacements[_SCIENCE_CANDIDATE_SITE_PATH])
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

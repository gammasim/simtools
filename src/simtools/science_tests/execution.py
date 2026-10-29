"""Prepare and execute science-test workflows for one site.

A workflow is resolved from its template: applications referencing the telescope
placeholder are repeated for each telescope of the site, placeholders are replaced, release
configuration overrides are applied, and the shared runtime environment is added. The
effective workflow is written to the test's work directory and executed from there.
"""

import copy
import json
import logging
import os
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import yaml

from simtools.constants import SCHEMA_PATH
from simtools.data_model import schema
from simtools.io.ascii_handler import to_builtin
from simtools.job_execution.execution import JobExecutionError
from simtools.production_configuration.compatibility import preview
from simtools.production_configuration.production_comparison import validate_comparison_inputs
from simtools.production_configuration.production_file_selection import normalize_for_comparison
from simtools.production_configuration.trigger_histograms import validate_derived_workflow
from simtools.runners.simtools_runner import (
    apply_execution_overrides,
    get_explicit_env_file,
    model_source_options,
    resolve_nested_configuration_files,
    run_applications,
)
from simtools.science_tests import definitions as defs
from simtools.science_tests.acceptance import AcceptanceError, evaluate_products
from simtools.science_tests.production import (
    archive_failed_production,
    check_inputs,
    collect_production,
)
from simtools.science_tests.reports import write_test_report
from simtools.utils import names
from simtools.utils.general import replace_placeholders_recursively
from simtools.version import __version__

EXECUTION_LOG = "execution.log"
WORKFLOW_RECORD = "workflow.yml"
STATUS = "status"
PASSED = {"pass", "warn"}
_APPLICATIONS = "applications"
_CONFIGURATION = "configuration"
_RUNTIME_ENVIRONMENT = "runtime_environment"
_IGNORE_RUNTIME = "ignore_runtime_environment"
_DERIVED_OUTPUT = "derived_output"
_PRODUCES_PRODUCTION = "produces_production"
_EXECUTION = "execution"
_REASON = "reason"
_TELESCOPES_KEY = "camera_telescopes"
_WORKFLOW_PATH = "workflow_path"
_LOG_FILE = "log_file"
_PROVENANCE_PATH = "provenance_path"
_WORKFLOW = "workflow"
_BASELINE = "baseline"
_RUNTIME = "runtime"
_OUTPUT_PATHS = "output_paths"
_PLANNED = "planned"
_COLLECT_PRODUCTION = "collect_production"
_PENDING = "pending"
_SUBMITTED = "submitted"
_IMAGE = "image"
_CONTAINER_IMAGE = "container_image"
# Settings that change on every execution and do not invalidate derived products.
_PER_EXECUTION_SETTINGS = {"activity_id", "disable_log_file", "label"}

logger = logging.getLogger(__name__)


@dataclass
class PreparedTest:
    """Resolved workflow and planned result of one site/test pair."""

    site: str
    test_id: str
    definition: dict
    replacements: dict
    args: dict
    workflow: dict
    work: Path
    report_root: Path
    result: dict


def prepare_test(bundle, site, test_id, application_args, run_id):
    """Resolve one site/test workflow without executing it.

    Parameters
    ----------
    bundle : ReleaseDefinition
        Release definitions.
    site, test_id : str
        Site and test names.
    application_args : dict
        Runner arguments including the model source.
    run_id : str
        Identifier shared by all tests of one runner call.

    Returns
    -------
    PreparedTest
        Prepared test with its planned result.
    """
    definition = defs.test_definition(bundle, test_id)
    if site not in definition["sites"]:
        raise ValueError(f"Test {test_id!r} does not support site {site!r}.")
    work = Path(bundle.context[defs.CANDIDATE_PATH]) / "work" / "science-tests" / site / test_id
    workflow_path = definition[_WORKFLOW_PATH]
    replacements = defs.site_replacements(bundle, site, work) | {
        defs.CONFIG_DIRECTORY: str(workflow_path.parent.resolve())
    }
    args = {
        _IGNORE_RUNTIME: False,
        "ignore_existing_parameter_version": False,
        **application_args,
        "steps": None,
        _LOG_FILE: str(work / EXECUTION_LOG),
        _PROVENANCE_PATH: str(work / WORKFLOW_RECORD),
    }
    workflow = _resolve_workflow(bundle, site, definition, replacements, args)
    runtime = None if args[_IGNORE_RUNTIME] else workflow.get(_RUNTIME_ENVIRONMENT)
    output_paths = [app[_CONFIGURATION].get("output_path") for app in workflow[_APPLICATIONS]]
    result = {
        "id": f"{test_id}.{site}",
        "test": test_id,
        "site": site,
        _WORKFLOW: str(definition[_WORKFLOW]),
        _BASELINE: bundle.release.get(_BASELINE),
        "candidate": bundle.release["release_label"],
        "execution_provenance": str(work),
        "report": None,
        "run_id": run_id,
        _RUNTIME: _runtime_description(runtime),
        "software": {
            "simtools": __version__,
            "simulation_models": args.get(defs.MODELS_GIT_REVISION),
        },
        _OUTPUT_PATHS: list(dict.fromkeys(filter(None, output_paths)))
        or [replacements[defs.CANDIDATE_SITE_PATH]],
        _EXECUTION: _PLANNED,
        STATUS: _PLANNED,
    }
    if definition.get(_PRODUCES_PRODUCTION):
        result["dependency_compatibility"] = _preview_production(
            workflow, args, replacements, runtime
        )
    return PreparedTest(
        site=site,
        test_id=test_id,
        definition=definition,
        replacements=replacements,
        args=args,
        workflow=workflow,
        work=work,
        report_root=bundle.release_dir / "reports" / site / test_id,
        result=result,
    )


def execute_tests(tests, catalogue, release_dir, run_id, overwrite=False):
    """Execute prepared tests in order, blocking tests whose selected dependencies failed.

    Parameters
    ----------
    tests : list of PreparedTest
        Prepared tests in dependency order.
    catalogue : dict
        Test definitions by name.
    release_dir : pathlib.Path
        Release directory for published reports.
    run_id : str
        Identifier used for archives.
    overwrite : bool
        Rebuild derived products and allow retrying failed productions.

    Returns
    -------
    list of dict
        Test results.
    """
    results = []
    for test in tests:
        result = test.result
        upstream = {
            f"{name}.{test.site}" for name in defs.dependency_order([test.test_id], catalogue)[:-1]
        }
        upstream_status = {r[STATUS] for r in results if r["id"] in upstream}
        if upstream_status - PASSED:
            status = _PENDING if upstream_status <= PASSED | {_SUBMITTED, _PENDING} else "blocked"
            results.append(
                result
                | {
                    STATUS: status,
                    _EXECUTION: status,
                    _REASON: "Dependency failed or is not complete.",
                }
            )
            continue
        _log_test(result, "Running")
        run_test(test, release_dir, run_id, overwrite)
        results.append(result)
    return results


def run_test(test, release_dir, run_id, overwrite=False):
    """Execute one prepared test, evaluate its products, and publish its report.

    Parameters
    ----------
    test : PreparedTest
        Prepared test.
    release_dir : pathlib.Path
        Release directory for published reports.
    run_id : str
        Identifier used for archives.
    overwrite : bool
        Rebuild derived products and allow retrying failed productions.
    """
    definition, replacements, result = test.definition, test.replacements, test.result
    result.update({_EXECUTION: "completed", STATUS: "pass"})
    try:
        if overwrite and definition.get(_PRODUCES_PRODUCTION):
            archive_failed_production(
                definition, replacements, run_id, (test.work, test.report_root)
            )
        check_inputs(definition, replacements)
        validate_comparison_inputs(
            test.workflow[_APPLICATIONS],
            [replacements[defs.CANDIDATE_SITE_PATH], replacements[defs.BASELINE_SITE_PATH]],
        )
        result.update(_execute(test, overwrite))
    except AcceptanceError as exc:
        result.update(status="warn", reason=f"Acceptance warning: {exc}")
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
        JobExecutionError,
    ) as exc:
        result.update(status="incomplete", execution="failed", error=str(exc))
    write_test_report(
        test.work,
        test.report_root,
        release_dir,
        result,
        definition.get("publish_report", True),
    )
    logger.info("%s: %s %s", result["id"], result[STATUS], result.get(_REASON, ""))


def log_planned(test):
    """Log the runtime and output paths of a planned test."""
    _log_test(test.result, "Planned")


def _log_test(result, action):
    logger.info("%s science test: %s", action, result["id"])
    logger.info("  Runtime: %s", result[_RUNTIME])
    logger.info("  Output: %s", ", ".join(result[_OUTPUT_PATHS]))


def _execute(test, overwrite):
    """Collect, reuse, or execute the workflow and return its evaluated status."""
    definition, replacements, args = test.definition, test.replacements, test.args
    if definition.get(_COLLECT_PRODUCTION):
        collected = collect_production(definition, replacements)
        if collected.get(STATUS) == _PENDING:
            return collected
        return evaluate_products(definition, replacements) | collected
    if not overwrite and _reusable_derived_product(test):
        return evaluate_products(definition, replacements) | {
            _EXECUTION: "reused",
            _REASON: "Reused complete derived products; use --overwrite to rebuild them.",
        }
    if definition.get(_DERIVED_OUTPUT):
        _remove(Path(replace_placeholders_recursively(definition[_DERIVED_OUTPUT], replacements)))
    _remove(test.work)
    args["config_file"] = str(_write_workflow(test.work, test.workflow))
    try:
        with _selected_model_source_environment(args):
            run_applications(args, replacements=replacements)
    finally:
        log_file = Path(args[_LOG_FILE])
        if log_file.is_file():
            test.result["execution_log"] = str(log_file)
    if definition.get(_DERIVED_OUTPUT):
        validate_derived_workflow(test.workflow[_APPLICATIONS])
    evaluated = evaluate_products(definition, replacements)
    if definition.get(_PRODUCES_PRODUCTION):
        return {
            STATUS: _SUBMITTED,
            _EXECUTION: _SUBMITTED,
            _REASON: "Jobs submitted; run the collection test after they finish.",
        }
    return evaluated


def _resolve_workflow(bundle, site, definition, replacements, args):
    """Expand, resolve, override, and validate one test workflow."""
    workflow = defs.load_yaml(definition[_WORKFLOW_PATH])
    telescopes = bundle.suites[site].get(_TELESCOPES_KEY, [])
    workflow[_APPLICATIONS] = [
        expanded
        for application in workflow.get(_APPLICATIONS, [])
        for expanded in _expand_telescopes(application, telescopes)
    ]
    workflow = replace_placeholders_recursively(workflow, replacements)
    allowed_changes = set(bundle.release.get("expected_changes", []))
    for application in workflow[_APPLICATIONS]:
        configuration = application.setdefault(_CONFIGURATION, {})
        configuration.update(definition["configuration_overrides"])
        undeclared = set(configuration.get("compare_by", [])) - allowed_changes
        if undeclared:
            raise ValueError(f"Comparison by {sorted(undeclared)} is not in expected_changes.")
    apply_execution_overrides(
        workflow[_APPLICATIONS], args, model_source_options(args), get_explicit_env_file(args)
    )
    production = definition.get(_PRODUCES_PRODUCTION) or definition.get(_COLLECT_PRODUCTION)
    if not production and _RUNTIME_ENVIRONMENT not in workflow:
        runtime = defs.shared_runtime_environment(
            bundle.template_dir, replacements, definition.get("runtime_role")
        )
        if runtime is not None:
            workflow[_RUNTIME_ENVIRONMENT] = runtime
    schema.validate_dict_using_schema(
        workflow, schema_file=SCHEMA_PATH / "application_workflow.metaschema.yml", offline=True
    )
    return workflow


def _expand_telescopes(application, telescopes):
    """Repeat an application for each telescope if it references the telescope placeholder."""
    if defs.TELESCOPE not in json.dumps(application):
        return [application]
    if not telescopes:
        raise ValueError(f"Workflow uses {defs.TELESCOPE}; set {_TELESCOPES_KEY} for the site.")
    return [
        replace_placeholders_recursively(
            application, {defs.TELESCOPE: names.validate_array_element_name(telescope)}
        )
        for telescope in telescopes
    ]


def _runtime_description(runtime):
    if runtime is None:
        return "host (no container)"
    return f"{runtime.get('container_engine', 'docker')} (image: {runtime.get(_IMAGE)})"


def _preview_production(workflow, args, replacements, runtime):
    """Check dependency compatibility of the production application and its image."""
    applications = copy.deepcopy(workflow[_APPLICATIONS][:1])
    resolve_nested_configuration_files(applications, replacements)
    selected = dict(args)
    image = runtime.get(_IMAGE) if runtime else None
    for application in applications:
        selected.update(application.get(_CONFIGURATION, {}))
        backend = selected.get("backend_config")
        if isinstance(backend, dict) and backend.get(_CONTAINER_IMAGE):
            image = backend[_CONTAINER_IMAGE]
    return preview(selected, image or replacements.get("__SCIENCE_CONTAINER_IMAGE_PATH__"))


def _reusable_derived_product(test):
    """Return whether existing derived products were produced with the current settings."""
    definition = test.definition
    if not definition.get(_DERIVED_OUTPUT):
        return False
    output = Path(replace_placeholders_recursively(definition[_DERIVED_OUTPUT], test.replacements))
    if not output.is_dir() or not any(output.iterdir()):
        return False
    try:
        previous = defs.load_yaml(Path(test.args[_PROVENANCE_PATH]))
        if _settings(previous) != _settings(test.workflow):
            raise ValueError("derivation settings changed")
        validate_derived_workflow(test.workflow[_APPLICATIONS])
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(
            f"Derived products are incomplete or stale ({exc}); rerun with --overwrite."
        ) from exc
    return True


def _settings(workflow):
    """Return application configurations without per-execution bookkeeping."""
    return normalize_for_comparison(
        [
            {
                key: value
                for key, value in app[_CONFIGURATION].items()
                if key not in _PER_EXECUTION_SETTINGS
            }
            for app in workflow[_APPLICATIONS]
        ]
    )


def _write_workflow(work, workflow):
    """Write the effective workflow executed by the runner."""
    work.mkdir(parents=True, exist_ok=True)
    path = work / WORKFLOW_RECORD
    path.write_text(
        yaml.safe_dump(to_builtin(workflow), sort_keys=False, explicit_start=True),
        encoding="utf-8",
    )
    return path


def _remove(path):
    if path.exists():
        shutil.rmtree(path)


@contextmanager
def _selected_model_source_environment(args):
    """Hide the inactive model source of a parent environment file from child applications."""
    inactive = None
    if args.get(defs.MODELS_GIT_PATH):
        inactive = "SIMTOOLS_SIMULATION_MODELS_PATH"
    elif args.get(defs.MODELS_PATH):
        inactive = "SIMTOOLS_SIMULATION_MODELS_GIT_PATH"
    previous = os.environ.get(inactive) if inactive else None
    if inactive:
        os.environ[inactive] = ""
    try:
        yield
    finally:
        if inactive and previous is None:
            os.environ.pop(inactive, None)
        elif inactive:
            os.environ[inactive] = previous

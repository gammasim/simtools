"""Small runner for reusable, site-parameterised release science tests."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

from simtools.constants import SCHEMA_PATH
from simtools.data_model import schema
from simtools.io import ascii_handler
from simtools.job_execution.job_manager import JobExecutionError
from simtools.runners.simtools_runner import prepare_workflow, run_applications
from simtools.utils.general import get_uuid, replace_placeholders_recursively

_VALID_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
_ROOT_KEYS = (
    "__SCIENCE_CANDIDATE_ROOT__",
    "__SCIENCE_BASELINE_ROOT__",
    "__SCIENCE_REFERENCE_ROOT__",
)


class _AcceptanceError(ValueError):
    """A valid measurement exceeds a configured acceptance limit."""


def _validate_release_context(release, context):
    """Validate explicit external roots and a consistent release identity."""
    label = release.get("release_label")
    if not isinstance(label, str) or not _VALID_NAME.fullmatch(label):
        raise ValueError("release_label must be a non-empty filename-safe label.")
    if context.get("__SCIENCE_RELEASE_LABEL__") != label:
        raise ValueError("Context and release labels differ.")
    for key in _ROOT_KEYS:
        value = context.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute() or "__" in value:
            raise ValueError(f"Context requires an absolute path for {key}.")
    if Path(context[_ROOT_KEYS[0]]).resolve() == Path(context[_ROOT_KEYS[1]]).resolve():
        raise ValueError("Candidate and baseline roots must differ.")


def _select(requested, available, kind):
    """Reject empty or unknown selections while removing duplicates."""
    selected = list(dict.fromkeys(requested if requested is not None else available))
    if not selected or set(selected) - set(available):
        raise ValueError(f"Invalid {kind} selection: {selected}; available: {sorted(available)}")
    return selected


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
            results.append(planned)
            continue
        dependencies = {f"{name}.{planned['site']}" for name in args[-1].get("depends_on", [])}
        if any(r["id"] in dependencies and r["status"] not in {"pass", "warn"} for r in results):
            results.append(
                {
                    **planned,
                    "status": "blocked",
                    "execution": "blocked",
                    "reason": "Dependency failed.",
                }
            )
        else:
            results.append(
                _run_test(*args, dry_run=False, run_id=run_id, application_args=application_args)
            )
    return results


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
        root = Path(replacements["__SCIENCE_CANDIDATE_SITE_ROOT__"])
        if any(root.rglob("submission.json")):
            raise ValueError(
                "Candidate already has a submission; do not resubmit. "
                "Resume it through the existing execution API or use a new candidate root."
            )
    if definition.get("requires_completed_production"):
        root = Path(replacements["__SCIENCE_CANDIDATE_SITE_ROOT__"])
        for path in _matched_files(str(root / "**" / "submission.json")):
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
        not Path(p).is_file() for paths in outputs.values() for p in paths
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
    release_dir, template_dir, release, catalogue, context, suites, products, rules
):
    """Identify definitions and external context without publishing absolute paths."""
    payload = [release, catalogue, context, suites, products, rules]
    for definition in catalogue.values():
        payload.append(
            _resolve_workflow(template_dir, release_dir, definition["workflow"]).read_text(
                encoding="utf-8"
            )
        )
    for profile in sorted((template_dir / "profiles").glob("*.yml")):
        payload.append(profile.read_text(encoding="utf-8"))
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
    release_dir: str | Path,
    *,
    context_file: str | Path | None = None,
    template_dir: str | Path | None = None,
    sites: Iterable[str] | None = None,
    tests: Iterable[str] | None = None,
    dry_run: bool = False,
    allow_production: bool = False,
    application_args: dict | None = None,
) -> dict:
    """Validate and run the selected site/test pairs for one release.

    The release directory contains only a small release.yml and the two site
    files. Test implementations are read from the shared template catalogue.

    Parameters
    ----------
    release_dir : str or pathlib.Path
        Directory containing release.yml and required site suites.
    context_file : str or pathlib.Path or None
        External candidate, baseline, reference roots and release label.
    template_dir : str or pathlib.Path or None
        Shared template; discovered above the release directory by default.
    sites, tests : iterable of str or None
        Selected sites/tests; comparison dependencies are included.
    dry_run : bool
        Validate the complete selection without writes or execution.
    allow_production : bool
        Permit explicitly selected production using a previously reviewed grid.
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
    release = _load_yaml(release_dir / "release.yml")
    template_dir = _resolve_template_dir(release_dir, template_dir)
    catalogue = _load_catalogue(release, release_dir, template_dir)
    context = _load_context(context_file)
    if not context:
        raise ValueError("A context_file is required for a science-test run.")
    if release.get("release_label"):
        context.setdefault("__SCIENCE_RELEASE_LABEL__", release["release_label"])
    _validate_release_context(release, context)
    suites = _load_suites(release_dir, release, catalogue)
    requested_sites = _select(sites, suites, "site")
    requested_tests = _select(tests, catalogue, "test") if tests is not None else None
    acceptance = _load_acceptance(release_dir, template_dir)
    signature = _campaign_signature(
        release_dir,
        template_dir,
        release,
        catalogue,
        context,
        suites,
        acceptance["products"],
        acceptance["rules"],
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
        failed_ids = ", ".join(result["id"] for result in failures)
        raise RuntimeError(f"Science tests failed: {failed_ids}")
    return summary


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
    report_root = release_dir / "reports" / "runs" / run_id / site / test_id
    work = (
        Path(context["__SCIENCE_CANDIDATE_ROOT__"])
        / "work"
        / "science-tests"
        / run_id
        / site
        / test_id
    )
    workflow = _resolve_workflow(template_dir, release_dir, definition["workflow"])
    replacements = _site_replacements(context, site, site_config, work)
    replacements["__CONFIG_DIRECTORY__"] = str(workflow.parent.resolve())
    resolved = replace_placeholders_recursively(
        _load_yaml(workflow),
        replacements,
    )
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
    prepare_workflow(args, replacements=replacements)

    result = {
        "id": f"{test_id}.{site}",
        "test": test_id,
        "site": site,
        "workflow": _relative_path(workflow, template_dir),
        "baseline": definition.get("baseline"),
        "candidate": context.get("__SCIENCE_RELEASE_LABEL__"),
        "execution_provenance": work.relative_to(
            Path(context["__SCIENCE_CANDIDATE_ROOT__"])
        ).as_posix(),
        "report": None,
        "run_id": run_id,
        "execution": "planned" if dry_run else "completed",
        "status": "planned" if dry_run else "pass",
    }
    if dry_run:
        return result

    try:
        _check_inputs(definition, replacements)
        run_applications(args, replacements=replacements)
        if definition.get("produces_production"):
            _check_inputs({"requires_completed_production": True}, replacements)
        result.update(_evaluate_products(definition, replacements))
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
    return result


def _write_test_report(work, report_root, release_dir, result):
    """Publish a small report even when execution or acceptance fails."""
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
        f"Baseline: {result['baseline']}; candidate: {result['candidate']}\n\n{reason}\n",
        encoding="utf-8",
    )


def _site_replacements(context, site, site_config, work):
    """Build placeholder values without requiring site-specific workflows."""
    site_label = site_config.get("site_label", site.title())
    candidate_root = Path(context.get("__SCIENCE_CANDIDATE_ROOT__", ""))
    baseline_root = Path(context.get("__SCIENCE_BASELINE_ROOT__", ""))
    replacements = dict(context)
    replacements.update(
        {
            "__SCIENCE_SITE_KEY__": site,
            "__SCIENCE_SITE__": site_label,
            "__SCIENCE_ARRAY_LAYOUT__": site_config["array_layout_name"],
            "__SCIENCE_CANDIDATE_SITE_ROOT__": str(candidate_root / site),
            "__SCIENCE_BASELINE_SITE_ROOT__": str(baseline_root / site),
            "__SCIENCE_REPORT_ROOT__": str(work / "output"),
            "__SCIENCE_COLLECTION_ROOT__": str(work / "collected"),
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
    if context_file is None:
        return {}
    context = _load_yaml(Path(context_file))
    if not isinstance(context, dict):
        raise ValueError("Science-test context must contain a mapping.")
    return context


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

"""Set up and run the science tests of one release."""

import logging
import re
from pathlib import Path

import yaml

from simtools.science_tests import definitions as defs
from simtools.science_tests.execution import execute_tests, log_planned, prepare_test
from simtools.science_tests.preflight import run_worker_canary, validate_context_paths
from simtools.science_tests.reports import REPORTS_DIRECTORY, release_summary, write_summary
from simtools.utils.general import get_uuid

_ENCODING = "utf-8"
_PRODUCES_PRODUCTION = "produces_production"
_FAILED = {"fail", "incomplete", "blocked"}
_STATUS = "status"
_SITES_DIRECTORY = "sites"

logger = logging.getLogger(__name__)


def run_release(
    release_dir,
    *,
    context_file=None,
    template_dir=None,
    sites=None,
    tests=None,
    dry_run=False,
    allow_production=False,
    preflight=False,
    overwrite=False,
    application_args=None,
):
    """Validate and run the selected site/test pairs of one release.

    Parameters
    ----------
    release_dir : str or pathlib.Path
        Directory containing release.yml and optional site overrides.
    context_file : str or pathlib.Path or None
        Context with candidate, baseline, and production-configuration paths;
        ``<release_dir>/context.yml`` by default.
    template_dir : str or pathlib.Path or None
        Shared template; discovered above the release directory by default.
    sites, tests : iterable of str or None
        Selected sites and tests; the required tests of each site by default. Dependencies
        selected together run first; otherwise their products must already exist.
    dry_run : bool
        Validate the complete selection without writes or execution.
    allow_production : bool
        Permit explicitly selected production tests after grid review.
    preflight : bool
        Check host paths and run one worker-side canary per site before production.
    overwrite : bool
        Rebuild derived products and archive failed, finished productions for a retry.
    application_args : dict or None
        Common runner arguments such as model-source and environment overrides.

    Returns
    -------
    dict
        Release summary with the result of every required site/test pair.

    Raises
    ------
    ValueError
        Invalid definitions, context, selection, or production permission.
    RuntimeError
        A selected test failed or is incomplete.
    """
    bundle = defs.load_release(release_dir, template_dir, context_file)
    selected_sites = defs.select(sites, bundle.suites, "site")
    selected_tests = None if tests is None else defs.select(tests, bundle.catalogue, "test")
    pairs = [
        (site, test_id)
        for site in selected_sites
        for test_id in _site_tests(bundle, site, selected_tests)
    ]
    production = [
        test_id for _, test_id in pairs if bundle.catalogue[test_id].get(_PRODUCES_PRODUCTION)
    ]
    if (
        production
        and not dry_run
        and not (allow_production and set(production) <= set(selected_tests or []))
    ):
        raise ValueError(
            f"Select production tests {sorted(set(production))} explicitly with --test and "
            "--allow_production after reviewing the generated grid."
        )
    defs.validate_context(
        bundle, any(defs.test_definition(bundle, test_id)["uses_baseline"] for _, test_id in pairs)
    )
    application_args = defs.model_source_arguments(application_args, bundle)
    run_id = get_uuid()
    prepared = [
        prepare_test(bundle, site, test_id, application_args, run_id) for site, test_id in pairs
    ]
    if dry_run:
        for test in prepared:
            log_planned(test)
        results = [test.result for test in prepared]
    else:
        if production:
            _preflight(bundle, prepared, preflight)
        results = execute_tests(prepared, bundle.catalogue, bundle.release_dir, run_id, overwrite)
    model_source = {
        key: str(application_args[key])
        for key in defs.MODEL_SOURCE_KEYS
        if application_args.get(key) is not None
    }
    summary = release_summary(
        bundle, results, dry_run, defs.campaign_signature(bundle, model_source)
    )
    if not dry_run:
        write_summary(bundle.release_dir / REPORTS_DIRECTORY, summary)
    failures = [result for result in results if result[_STATUS] in _FAILED]
    if failures:
        details = "; ".join(
            f"{r['id']}: {r.get('error') or r.get('reason') or r[_STATUS]}" for r in failures
        )
        raise RuntimeError(f"Science tests failed: {details}")
    return summary


def _site_tests(bundle, site, selected_tests):
    """Return the selected (or required) tests of a site in dependency order."""
    tests = bundle.suites[site]["required_tests"] if selected_tests is None else selected_tests
    return [
        test_id for test_id in defs.dependency_order(tests, bundle.catalogue) if test_id in tests
    ]


def _preflight(bundle, prepared, preflight):
    """Check host paths and run one worker canary per site with production tests."""
    if not preflight:
        logger.warning(
            "Production science tests are submitted without --preflight; "
            "host and worker filesystem checks were skipped."
        )
        return
    validate_context_paths(bundle.context)
    canaries = {}
    for test in prepared:
        if test.definition.get(_PRODUCES_PRODUCTION):
            canaries.setdefault(test.site, test)
    for site, test in canaries.items():
        work_dir = test.work.parents[1] / "preflight" / site
        run_worker_canary(test, bundle.context, work_dir)


def setup_release(
    release_dir,
    template_dir=None,
    context_file=None,
    *,
    baseline_version=None,
    simulation_models_git_revision=None,
):
    """Create a release directory from the shared templates.

    Copies ``release.yml``, all site templates, and the context template, replacing the
    release-label placeholder by the name of the release directory's parent.

    Parameters
    ----------
    release_dir : str or pathlib.Path
        Destination directory.
    template_dir : str or pathlib.Path or None
        Shared template; discovered above the release directory by default.
    context_file : str or pathlib.Path or None
        Context template; ``context.example.yml`` of the template by default.
    baseline_version : str or None
        Baseline release label; also the last component of the baseline context path.
    simulation_models_git_revision : str or None
        Model Git revision recorded in the release definition.

    Returns
    -------
    pathlib.Path
        Created release directory.
    """
    if baseline_version is not None and not defs.VALID_NAME.fullmatch(baseline_version):
        raise ValueError(f"baseline_version must be a filename-safe label: {baseline_version!r}")
    release_dir = Path(release_dir).resolve()
    template_dir = defs.resolve_template_dir(release_dir, template_dir)
    release_label = release_dir.parent.name
    if not defs.VALID_NAME.fullmatch(release_label):
        release_label = "candidate"
    site_templates = template_dir / _SITES_DIRECTORY
    sites = sorted(site_templates.glob("*.yml"))
    if not sites:
        raise FileNotFoundError(f"Science-test site templates not found: {site_templates}")
    files = {release_dir / defs.RELEASE_FILE: template_dir / defs.RELEASE_FILE}
    files |= {release_dir / _SITES_DIRECTORY / site.name: site for site in sites}
    files[release_dir / defs.CONTEXT_FILE] = (
        Path(context_file) if context_file else template_dir / "context.example.yml"
    )
    existing = [str(target) for target in files if target.exists()]
    if existing:
        raise FileExistsError(f"Science-test setup files already exist: {', '.join(existing)}")
    missing = [str(source) for source in files.values() if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"Science-test setup templates not found: {', '.join(missing)}")
    overrides = {
        defs.RELEASE_FILE: {
            "baseline": baseline_version,
            defs.MODELS_GIT_REVISION: simulation_models_git_revision,
        },
        defs.CONTEXT_FILE: {defs.BASELINE_PATH: baseline_version},
    }
    rendered = {
        target: _render(source, release_label, overrides.get(target.name, {}))
        for target, source in files.items()
    }
    for target, contents in rendered.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents, encoding=_ENCODING)
    logger.info("Created science-test setup in %s", release_dir)
    logger.info("Edit %s and review %s before the dry run.", defs.CONTEXT_FILE, defs.RELEASE_FILE)
    return release_dir


def _render(source, release_label, overrides):
    """Return a template with the release label and the given top-level values replaced.

    Lines are replaced in place to keep template comments. The baseline context path keeps
    its parent directory and replaces only the release directory name.
    """
    contents = source.read_text(encoding=_ENCODING).replace(defs.RELEASE_LABEL, release_label)
    for key, value in overrides.items():
        if value is None:
            continue
        if key == defs.BASELINE_PATH:
            baseline = (yaml.safe_load(contents) or {}).get(key)
            if not baseline:
                raise ValueError(f"Setup context requires {key} to set the baseline.")
            value = str(Path(baseline).parent / value)
        line = yaml.safe_dump({key: value}, sort_keys=False).rstrip()
        pattern = rf"^{re.escape(key)}:.*$"
        if re.search(pattern, contents, flags=re.MULTILINE):
            contents = re.sub(pattern, lambda _match, line=line: line, contents, flags=re.MULTILINE)
        else:
            contents = f"{contents.rstrip()}\n{line}\n"
    return contents if contents.startswith("---") else f"---\n{contents}"

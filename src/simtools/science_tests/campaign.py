"""Coordinate release science-test runs."""

import logging
from pathlib import Path

from simtools.science_tests.definitions import (
    _MODEL_SOURCE_KEYS,
    _RELEASE_FILE_NAME,
    _SETUP_RELEASE_LABEL,
    _VALID_NAME,
    _campaign_signature,
    _check_production_selection,
    _dependency_order,
    _load_acceptance,
    _load_catalogue,
    _load_context,
    _load_suites,
    _load_yaml,
    _model_source_arguments,
    _resolve_template_dir,
    _select,
    _test_definition,
    _validate_definition,
    _validate_selected_context,
)
from simtools.science_tests.execution import _execute_selection, _run_test
from simtools.science_tests.reports import (
    _release_summary,
    _write_summary,
)
from simtools.utils.general import get_uuid

logger = logging.getLogger(__name__)


def _prepare_release_run(
    release_dir, template_dir, context_file, sites, tests, overwrite, application_args
):
    """Load and validate the shared state required for a release run."""
    release_dir = Path(release_dir).resolve()
    release = _load_yaml(release_dir / _RELEASE_FILE_NAME)
    template_dir = _resolve_template_dir(release_dir, template_dir)
    catalogue = _load_catalogue(release, release_dir, template_dir)
    context = _load_context(context_file)
    if not context:
        raise ValueError("A context_file is required for a science-test run.")
    context["__SCIENCE_RELEASE_LABEL__"] = release["release_label"]
    suites = _load_suites(release_dir, release, catalogue, template_dir)
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
        if application_args.get(key) is not None
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
    return {
        "release_dir": release_dir,
        "release": release,
        "template_dir": template_dir,
        "catalogue": catalogue,
        "context": context,
        "suites": suites,
        "requested_sites": requested_sites,
        "requested_tests": requested_tests,
        "acceptance": acceptance,
        "application_args": application_args,
        "signature": signature,
    }


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
        Selected sites/tests. Only selected tests are executed; dependencies are ordered when
        selected together and otherwise their required products must already exist.
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
    run = _prepare_release_run(
        release_dir, template_dir, context_file, sites, tests, overwrite, application_args
    )
    prepared = []
    run_id = get_uuid()
    for site in run["requested_sites"]:
        site_config = run["suites"][site]
        selected = (
            site_config["required_tests"]
            if run["requested_tests"] is None
            else run["requested_tests"]
        )
        ordered = _dependency_order(selected, run["catalogue"])
        for test_id in (name for name in ordered if name in selected):
            _validate_definition(test_id, site, run["catalogue"][test_id])
            definition = _test_definition(
                test_id, run["catalogue"], run["acceptance"], run["release"]
            )
            _check_production_selection(
                test_id, definition, run["requested_tests"], allow_production, dry_run
            )
            args = (
                run["release_dir"],
                run["template_dir"],
                run["context"],
                site,
                site_config,
                test_id,
                definition,
            )
            prepared.append(
                (
                    args,
                    _run_test(
                        *args,
                        dry_run=True,
                        run_id=run_id,
                        application_args=run["application_args"],
                    ),
                )
            )
    results = _execute_selection(
        prepared, dry_run, run_id, run["application_args"], run["catalogue"]
    )
    summary = _release_summary(
        run["release_dir"],
        run["release"],
        run["suites"],
        results,
        dry_run,
        run["signature"],
        run["catalogue"],
    )
    if not dry_run:
        _write_summary(run["release_dir"] / "reports", summary)
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

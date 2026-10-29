"""Check, collect, and retry science-test productions.

Production tests submit jobs and record a submission manifest below the candidate site
directory. Collection waits for those jobs and reconciles their outputs with the reviewed
job grid before downstream tests use them.
"""

import json
import logging
from collections import Counter
from pathlib import Path

from simtools.io.file_type import product_filename
from simtools.job_execution.backends.registry import get_backend
from simtools.job_execution.execution import collect_submission, load_submission
from simtools.production_configuration.jobs_io import read_job_grid
from simtools.production_configuration.production_file_selection import (
    discover_manifests,
    normalize_for_comparison,
)
from simtools.science_tests.acceptance import matched_files, nonempty
from simtools.science_tests.definitions import (
    BASELINE_PATH,
    BASELINE_SITE_PATH,
    CANDIDATE_PATH,
    CANDIDATE_SITE_PATH,
)
from simtools.utils.value_conversion import replace_placeholders_recursively

SUBMISSION_FILE = product_filename("submission_manifest")
_ENCODING = "utf-8"
_JOB_IDS = "job_ids"
_STATE = "state"
_HTCONDOR = "htcondor"
_REQUIRES = "requires"
_REQUIRES_COMPLETED_PRODUCTION = "requires_completed_production"
_REVIEWED_GRID = "reviewed_grid"
_COMPLETED = "completed"
_PENDING = "pending"
_JOB_ID = "job_id"
_ARCHIVE = "archive"
# Grid columns that do not define the simulated configuration.
_GRID_FIELDS_NOT_COMPARED = {
    "nsb_rate",
    "overwrite_model_parameters",
    "scan_label",
    "model_parameter_set",
}

logger = logging.getLogger(__name__)


def check_inputs(definition, replacements):
    """Require declared inputs and, where configured, complete reviewed productions.

    Parameters
    ----------
    definition : dict
        Test definition.
    replacements : dict
        Placeholder values.
    """
    for pattern in definition.get(_REQUIRES, []):
        matched_files(replace_placeholders_recursively(pattern, replacements))
    site_path = Path(replacements[CANDIDATE_SITE_PATH])
    if definition.get("produces_production"):
        submission = next(site_path.rglob(SUBMISSION_FILE), None)
        if submission is not None:
            raise ValueError(
                f"Production submission blocked by existing record {submission}. "
                "Use --overwrite to archive a failed production once all jobs have ended, "
                "or use a new candidate path."
            )
    if definition.get(_REVIEWED_GRID) and not definition.get("collect_production"):
        check_reviewed_production(definition, replacements)
        if definition.get("uses_baseline"):
            check_reviewed_production(definition, _baseline_replacements(replacements))
    if definition.get(_REQUIRES_COMPLETED_PRODUCTION):
        for path in matched_files(str(site_path / "**" / SUBMISSION_FILE)):
            _check_completed_submission(path)


def collect_production(definition, replacements):
    """Collect finished jobs of all submissions and validate completed productions.

    Parameters
    ----------
    definition : dict
        Collection test definition.
    replacements : dict
        Placeholder values.

    Returns
    -------
    dict
        ``pending`` result while jobs are queued, otherwise the production coverage.
    """
    site_path = Path(replacements[CANDIDATE_SITE_PATH])
    pending = False
    for manifest in matched_files(str(site_path / "**" / SUBMISSION_FILE)):
        submission = load_submission(manifest)
        if submission.backend != _HTCONDOR:
            raise ValueError(f"Collection is not supported for backend {submission.backend}.")
        if submission.metadata.get(_STATE) != _COMPLETED:
            pending = collect_submission(submission) is None or pending
    if pending:
        return {
            "status": _PENDING,
            "execution": _PENDING,
            "reason": "Production jobs are still queued; rerun collection after they finish.",
        }
    check_inputs({_REQUIRES_COMPLETED_PRODUCTION: True}, replacements)
    if definition.get(_REVIEWED_GRID):
        return {"coverage": check_reviewed_production(definition, replacements)}
    return {}


def archive_failed_production(definition, replacements, run_id, paths):
    """Archive a failed, finished production and its reports before a retry.

    Parameters
    ----------
    definition : dict
        Production test definition.
    replacements : dict
        Placeholder values.
    run_id : str
        Identifier used as archive directory name.
    paths : iterable of pathlib.Path
        Work and report directories archived together with the site production.
    """
    site_path = Path(replacements[CANDIDATE_SITE_PATH])
    submissions = list(site_path.rglob(SUBMISSION_FILE))
    if not submissions:
        return
    for path in definition.get(_REQUIRES, []):
        matched_files(replace_placeholders_recursively(path, replacements))
    for manifest in submissions:
        submission = load_submission(manifest)
        if submission.metadata.get(_STATE) != "failed":
            reason = "only failed production can be retried"
        elif submission.backend != _HTCONDOR or not get_backend(submission.backend).is_finished(
            submission
        ):
            reason = "jobs may still be queued"
        else:
            continue
        raise ValueError(f"Cannot overwrite {manifest}: {reason}.")
    archives = [(site_path, site_path.parent / _ARCHIVE / run_id / site_path.name)]
    archives += [
        (path, path.parents[1] / _ARCHIVE / run_id / path.parent.name / path.name) for path in paths
    ]
    for source, destination in archives:
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            logger.info("Archived previous run: %s", destination)


def check_reviewed_production(definition, replacements):
    """Reconcile completed jobs and their outputs with the reviewed job grid.

    Parameters
    ----------
    definition : dict
        Test definition with ``reviewed_grid``.
    replacements : dict
        Placeholder values.

    Returns
    -------
    dict
        Expected and observed job counts.
    """
    rows, metadata = read_job_grid(
        replace_placeholders_recursively(definition[_REVIEWED_GRID], replacements)
    )
    site_path = Path(replacements[CANDIDATE_SITE_PATH])
    discovered = discover_manifests(site_path)
    expected = {f"job-{index:06d}" for index in range(1, len(rows) + 1)}
    _require_same_ids(expected, [item.data.get(_JOB_ID) for item in discovered], "production jobs")
    manifests = {item.data[_JOB_ID]: item for item in discovered}
    fixed = {key: metadata[key] for key in ("site", "simulation_software") if key in metadata}
    for index, row in enumerate(rows, start=1):
        manifest = manifests[f"job-{index:06d}"]
        _check_grid_row(row | fixed, manifest.data["configuration"])
        _check_job_outputs(manifest)
    submitted = []
    for path in matched_files(str(site_path / "**" / SUBMISSION_FILE)):
        submitted.extend(_check_completed_submission(path))
    _require_same_ids(expected, submitted, "submitted jobs")
    return {"expected_jobs": len(rows), "observed_jobs": len(manifests)}


def _check_job_outputs(manifest):
    """Require reduced event data and non-empty outputs inside the job directory."""
    files = manifest.data["files"]
    if not files.get("reduced_event_data"):
        raise ValueError(f"Missing reduced event data in {manifest.path}.")
    root = manifest.directory.resolve()
    for relative in (path for paths in files.values() for path in paths):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not nonempty(path):
            raise ValueError(f"Missing or empty production output: {path}")


def _check_completed_submission(path):
    """Require a completed submission whose jobs all produced their declared outputs."""
    payload = json.loads(path.read_text(encoding=_ENCODING))
    metadata = payload.get("metadata", {})
    if metadata.get(_STATE) != _COMPLETED:
        raise ValueError(f"Production is not completed: {path}")
    jobs = payload.get(_JOB_IDS, [])
    outputs = metadata.get("expected_outputs", {})
    if (
        not jobs
        or set(jobs) != set(outputs)
        or not all(paths and all(nonempty(Path(p)) for p in paths) for paths in outputs.values())
    ):
        raise ValueError(f"Production has missing expected outputs: {path}")
    return jobs


def _check_grid_row(row, configuration):
    """Compare the simulation-defining values of a grid row with a job configuration."""
    for key, value in row.items():
        if key in _GRID_FIELDS_NOT_COMPARED:
            continue
        observed = configuration.get(key)
        if isinstance(observed, list) and len(observed) == 1 and not isinstance(value, list):
            observed = observed[0]
        if normalize_for_comparison(value) != normalize_for_comparison(observed):
            raise ValueError(
                f"Reviewed grid mismatch for run {row.get('run_number')}, {key}: "
                f"expected {value}, observed {observed}."
            )


def _require_same_ids(expected, observed, description):
    """Reject missing, unexpected, and duplicate job identifiers."""
    counts = Counter(observed)
    missing = sorted(expected - counts.keys())
    extra = sorted(str(key) for key in counts.keys() - expected)
    duplicate = sorted(str(key) for key, count in counts.items() if count > 1)
    if missing or extra or duplicate:
        raise ValueError(
            f"Incomplete {description}: missing={missing[:5]}, extra={extra[:5]}, "
            f"duplicate={duplicate[:5]}."
        )


def _baseline_replacements(replacements):
    """Point candidate placeholders to the baseline production."""
    return replacements | {
        CANDIDATE_PATH: replacements[BASELINE_PATH],
        CANDIDATE_SITE_PATH: replacements[BASELINE_SITE_PATH],
    }

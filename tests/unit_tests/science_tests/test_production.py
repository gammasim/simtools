import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from simtools.science_tests import definitions as defs
from simtools.science_tests import production

NORTH = "North"
ARCHIVE = "archive"
BASELINE = "baseline"
CANDIDATE = "candidate"
CONFIGURATION = "configuration"
EVENTS_DAT = "events.dat"
EXPECTED_OUTPUTS = "expected_outputs"
FILES = "files"
LOAD_SUBMISSION = "load_submission"
NORTH_SITE = "north"
PRODUCTION = "production"
REDUCED_EVENT_DATA = "reduced_event_data"
REQUIRES_COMPLETED_PRODUCTION = "requires_completed_production"
REVIEWED_GRID = "reviewed_grid"
RUN_NUMBER = "run_number"
SUBMITTED = "submitted"
UTF_8 = "utf-8"
ZENITH_ANGLE = "zenith_angle"

JOB = "job-000001"
STATE = "state"
METADATA = "metadata"
HTCONDOR = "htcondor"
GRID = "grid.ecsv"
COMPLETED = "completed"


@pytest.fixture
def site(tmp_test_directory):
    root = Path(tmp_test_directory) / PRODUCTION
    replacements = {
        defs.CANDIDATE_PATH: str(root / CANDIDATE),
        defs.CANDIDATE_SITE_PATH: str(root / CANDIDATE / NORTH_SITE),
        defs.BASELINE_PATH: str(root / BASELINE),
        defs.BASELINE_SITE_PATH: str(root / BASELINE / NORTH_SITE),
    }
    for key in (defs.CANDIDATE_SITE_PATH, defs.BASELINE_SITE_PATH):
        Path(replacements[key]).mkdir(parents=True)
    return replacements


def write_submission(site_path, state=COMPLETED, output=b"data"):
    job = Path(site_path) / JOB
    job.mkdir(parents=True, exist_ok=True)
    (job / EVENTS_DAT).write_bytes(output)
    path = Path(site_path) / "htcondor-1" / production.SUBMISSION_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_ids": [JOB],
        METADATA: {STATE: state, EXPECTED_OUTPUTS: {JOB: [str(job / EVENTS_DAT)]}},
    }
    path.write_text(json.dumps(payload), encoding=UTF_8)
    return path, payload


def mock_grid(mocker, site_path, configuration, files=None):
    directory = Path(site_path) / JOB
    manifest = SimpleNamespace(
        data={
            "job_id": JOB,
            CONFIGURATION: configuration,
            FILES: files or {REDUCED_EVENT_DATA: [EVENTS_DAT]},
        },
        directory=directory,
        path=directory / "manifest.yml",
    )
    mocker.patch.object(
        production,
        "read_job_grid",
        return_value=([{RUN_NUMBER: 1, ZENITH_ANGLE: 20, "nsb_rate": 1}], {"site": NORTH}),
    )
    mocker.patch.object(production, "discover_manifests", return_value=[manifest])


def test_check_inputs_blocks_existing_submission(site):
    write_submission(site[defs.CANDIDATE_SITE_PATH])
    with pytest.raises(ValueError, match="Production submission blocked"):
        production.check_inputs({"produces_production": True}, site)
    production.check_inputs({REQUIRES_COMPLETED_PRODUCTION: True}, site)


def test_check_inputs_requires_completed_submission(site):
    path, payload = write_submission(site[defs.CANDIDATE_SITE_PATH], state=SUBMITTED)
    with pytest.raises(ValueError, match="not completed"):
        production.check_inputs({REQUIRES_COMPLETED_PRODUCTION: True}, site)
    payload[METADATA][STATE] = COMPLETED
    payload[METADATA][EXPECTED_OUTPUTS][JOB] = []
    path.write_text(json.dumps(payload), encoding=UTF_8)
    with pytest.raises(ValueError, match="missing expected outputs"):
        production.check_inputs({REQUIRES_COMPLETED_PRODUCTION: True}, site)


def test_check_reviewed_production_matches_grid(site, mocker):
    write_submission(site[defs.CANDIDATE_SITE_PATH])
    write_submission(site[defs.BASELINE_SITE_PATH])
    mock_grid(
        mocker,
        site[defs.CANDIDATE_SITE_PATH],
        {ZENITH_ANGLE: [20], "site": NORTH, RUN_NUMBER: 1},
    )
    definition = {REVIEWED_GRID: GRID, "uses_baseline": True}

    production.check_inputs(definition, site)
    coverage = production.check_reviewed_production(definition, site)

    assert coverage == {"expected_jobs": 1, "observed_jobs": 1}


@pytest.mark.parametrize(
    (CONFIGURATION, FILES, "message"),
    [
        ({ZENITH_ANGLE: 40, "site": NORTH}, None, "Reviewed grid mismatch"),
        (
            {ZENITH_ANGLE: 20, "site": NORTH, RUN_NUMBER: 1},
            {"other": []},
            "Missing reduced event data",
        ),
        (
            {ZENITH_ANGLE: 20, "site": NORTH, RUN_NUMBER: 1},
            {REDUCED_EVENT_DATA: ["../outside.dat"]},
            "Missing or empty production output",
        ),
    ],
)
def test_check_reviewed_production_rejects_mismatches(site, mocker, configuration, files, message):
    write_submission(site[defs.CANDIDATE_SITE_PATH])
    mock_grid(mocker, site[defs.CANDIDATE_SITE_PATH], configuration, files)
    with pytest.raises(ValueError, match=message):
        production.check_reviewed_production({REVIEWED_GRID: GRID}, site)


def test_check_reviewed_production_rejects_missing_jobs(site, mocker):
    write_submission(site[defs.CANDIDATE_SITE_PATH])
    mock_grid(mocker, site[defs.CANDIDATE_SITE_PATH], {})
    production.discover_manifests.return_value = []
    with pytest.raises(ValueError, match="Incomplete production jobs"):
        production.check_reviewed_production({REVIEWED_GRID: GRID}, site)


def test_collect_production_reports_pending_and_completed(site, mocker):
    path, payload = write_submission(site[defs.CANDIDATE_SITE_PATH], state=SUBMITTED)
    submission = SimpleNamespace(backend=HTCONDOR, metadata=payload[METADATA])
    mocker.patch.object(production, LOAD_SUBMISSION, return_value=submission)
    collect = mocker.patch.object(production, "collect_submission", return_value=None)

    assert production.collect_production({}, site)["status"] == "pending"

    collect.return_value = object()
    payload[METADATA][STATE] = COMPLETED
    path.write_text(json.dumps(payload), encoding=UTF_8)
    assert production.collect_production({}, site) == {}
    mocker.patch.object(production, "check_reviewed_production", return_value={"jobs": 1})
    assert production.collect_production({REVIEWED_GRID: GRID}, site) == {"coverage": {"jobs": 1}}
    submission.backend = "local"
    with pytest.raises(ValueError, match="not supported for backend local"):
        production.collect_production({}, site)


def test_archive_failed_production(site, mocker, tmp_test_directory):
    site_path = Path(site[defs.CANDIDATE_SITE_PATH])
    production.archive_failed_production({}, site, "run", [])
    _, payload = write_submission(site_path, state="failed")
    submission = SimpleNamespace(backend=HTCONDOR, metadata=payload[METADATA])
    mocker.patch.object(production, LOAD_SUBMISSION, return_value=submission)
    backend = mocker.patch.object(production, "get_backend")
    backend.return_value.is_finished.return_value = False
    with pytest.raises(ValueError, match="may still be queued"):
        production.archive_failed_production({}, site, "run", [])

    backend.return_value.is_finished.return_value = True
    work = Path(tmp_test_directory) / PRODUCTION / "work" / NORTH_SITE / "test"
    work.mkdir(parents=True)
    production.archive_failed_production({}, site, "run", [work])

    assert not site_path.exists()
    assert (site_path.parent / ARCHIVE / "run" / NORTH_SITE / JOB).is_dir()
    assert (work.parents[1] / ARCHIVE / "run" / NORTH_SITE / "test").is_dir()


def test_archive_rejects_unfailed_production(site, mocker):
    _, payload = write_submission(site[defs.CANDIDATE_SITE_PATH])
    submission = SimpleNamespace(backend=HTCONDOR, metadata=payload[METADATA])
    mocker.patch.object(production, LOAD_SUBMISSION, return_value=submission)
    with pytest.raises(ValueError, match="only failed production"):
        production.archive_failed_production({}, site, "run", [])


def test_require_same_ids_reports_duplicates():
    with pytest.raises(ValueError, match="duplicate=\\['a'\\]"):
        production._require_same_ids({"a"}, ["a", "a"], "jobs")

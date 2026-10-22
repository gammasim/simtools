from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from simtools.science_tests import preflight

DATA = "__DATA__"
IMAGE = "__IMAGE__"
MISSING = "__MISSING__"
BACKEND_YML = "backend.yml"
BACKEND_CONFIG = "backend_config"
BACKEND_CONFIG_WITH_CONTAINER_IMAGE = "backend_config with container_image"
EMPTY = "empty"
PRIORITY = "priority"

ENCODING = "utf-8"


@pytest.fixture
def paths(tmp_test_directory):
    root = Path(tmp_test_directory) / "preflight"
    directory = root / "data"
    directory.mkdir(parents=True)
    (directory / "file.txt").write_text("x", encoding=ENCODING)
    image = root / "image.sif"
    image.write_text("image", encoding=ENCODING)
    return root, {DATA: str(directory), IMAGE: str(image), "label": "v1", "none": None}


def test_validate_context_paths_checks_absolute_paths(paths):
    root, context = paths
    preflight.validate_context_paths(context)
    (root / EMPTY).mkdir()
    context |= {"__EMPTY__": str(root / EMPTY), MISSING: str(root / "missing")}
    with pytest.raises(ValueError, match="__EMPTY__ is missing or empty") as error:
        preflight.validate_context_paths(context)
    assert MISSING in str(error.value)


def prepared_test(root, backend_config):
    return SimpleNamespace(
        result={"id": "production.north"},
        replacements={"__CONFIG_DIRECTORY__": str(root)},
        workflow={
            "applications": [
                {"application": "simtools-simulate-prod", "configuration": backend_config}
            ]
        },
    )


def test_worker_canary_checks_context_and_environment_paths(paths, mocker):
    root, context = paths
    environment = root / "job.env"
    environment.write_text(
        "SIMTOOLS_CORSIKA_PATH=/workdir/corsika\nSIMTOOLS_OTHER_PATH=${ROOT}/x\nNAME=value\n"
        'export SIMTOOLS_SIM_TELARRAY_PATH="/workdir/sim_telarray"\n',
        encoding=ENCODING,
    )
    backend = root / BACKEND_YML
    backend.write_text(
        yaml.safe_dump({"container_image": context[IMAGE], "environment_file": str(environment)}),
        encoding=ENCODING,
    )
    submit = mocker.patch.object(preflight, "submit_jobs", return_value="submission")
    wait = mocker.patch.object(preflight, "wait_for_submission")

    preflight.run_worker_canary(
        prepared_test(root, {BACKEND_CONFIG: BACKEND_YML}), context, root / "work"
    )

    job, options = submit.call_args.args[0][0], submit.call_args.args[1]
    script = job.command[2]
    assert options.backend_config[PRIORITY] == preflight.PREFLIGHT_PRIORITY
    assert f"check_path __DATA__ {context[DATA]}" in script
    assert "check_path SIMTOOLS_CORSIKA_PATH /workdir/corsika" in script
    assert "check_path SIMTOOLS_SIM_TELARRAY_PATH /workdir/sim_telarray" in script
    assert "OTHER" not in script
    assert IMAGE not in script
    wait.assert_called_once()


def test_worker_canary_requires_container_backend(paths):
    root, context = paths
    with pytest.raises(ValueError, match=BACKEND_CONFIG_WITH_CONTAINER_IMAGE):
        preflight.run_worker_canary(prepared_test(root, {}), context, root)
    with pytest.raises(ValueError, match=BACKEND_CONFIG_WITH_CONTAINER_IMAGE):
        preflight.run_worker_canary(
            prepared_test(root, {BACKEND_CONFIG: {PRIORITY: 1}}), context, root
        )

#!/usr/bin/python3

import copy
import logging
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import astropy.units as u
import pytest

from simtools import settings
from simtools.runners import runner_services
from simtools.runners.corsika_runner import CorsikaRunner
from simtools.runners.corsika_simtel_runner import CorsikaSimtelRunner, _shower_route_seeds
from simtools.simtel.simulator_array import SimulatorArray

logger = logging.getLogger()


@pytest.fixture
def simtel_command():
    """Basic simtel command."""
    return "bin/sim_telarray"


@pytest.fixture
def show_all():
    """Simtel show all options."""
    return "-C show=all"


@pytest.fixture
def simulation_file(model_version):
    """Base name for simulation test file."""
    return (
        f"proton_run000001_za20deg_azm000deg_South_test_layout_{model_version}_"
        "test-corsika-simtel-runner"
    )


@pytest.fixture
def corsika_simtel_runner(corsika_config_mock_array_model):
    """CorsikaSimtelRunner object."""
    return CorsikaSimtelRunner(
        corsika_config=corsika_config_mock_array_model,
        label="test-corsika-simtel-runner",
    )


def test_corsika_simtel_runner(corsika_simtel_runner):
    assert isinstance(corsika_simtel_runner.corsika_runner, CorsikaRunner)
    assert isinstance(corsika_simtel_runner.simulator_array[0], SimulatorArray)


def test_get_resources(corsika_simtel_runner, mocker):
    get_resources = mocker.patch.object(
        corsika_simtel_runner.corsika_runner,
        "get_resources",
        return_value={"runtime": 2.5, "n_events": 100},
    )

    result = corsika_simtel_runner.get_resources(runtime=2.5)

    assert result == {"runtime": 2.5, "n_events": 100}
    get_resources.assert_called_once_with(runtime=2.5)


def test_update_file_list_flattens_repeated_resource_files(corsika_simtel_runner, mocker):
    corsika_simtel_runner.file_list = {}
    corsika_simtel_runner.corsika_runner.file_list = {}
    first_array = mocker.Mock(file_list={"sim_telarray_resources": ["first", "second"]})
    second_array = mocker.Mock(file_list={"sim_telarray_resources": "third"})
    corsika_simtel_runner.simulator_array = [first_array, second_array]

    corsika_simtel_runner.update_file_list_from_runners()

    assert corsika_simtel_runner.file_list["sim_telarray_resources"] == [
        "first",
        "second",
        "third",
    ]


def test_prepare_run(corsika_simtel_runner, tmp_path):
    # prepare_run now requires sub_script parameter and doesn't return the script path
    script_path = tmp_path / "test_script.sh"
    corsika_simtel_runner.prepare_run(run_number=1, sub_script=script_path)

    assert script_path.exists()
    with open(script_path) as f:
        script_content = f.read()
        assert "/usr/bin/env bash" in script_content

    # Run number is given
    run_number = 3
    script_path2 = tmp_path / "test_script2.sh"
    corsika_simtel_runner.prepare_run(run_number=run_number, sub_script=script_path2)

    assert script_path2.exists()
    with open(script_path2) as f:
        script_content = f.read()
        assert "/usr/bin/env bash" in script_content


def test_export_multipipe_script(corsika_simtel_runner, simtel_command, show_all):
    corsika_simtel_runner._export_multipipe_script(run_number=1)
    script = corsika_simtel_runner.runner_service.get_file_name("multi_pipe_config", run_number=1)

    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        assert simtel_command in script_content
        assert (
            f"--model-version {corsika_simtel_runner.base_corsika_config.array_model.model_version}"
            in script_content
        )
        assert "-C telescope_theta=20" in script_content
        assert "-C telescope_phi=0" in script_content
        assert show_all in script_content
        assert "zstd >" not in script_content

    # Test second call
    corsika_simtel_runner._export_multipipe_script(run_number=1)
    script = corsika_simtel_runner.runner_service.get_file_name("multi_pipe_config", run_number=1)

    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        # For calibration runs, we should see noise settings
        if corsika_simtel_runner.base_corsika_config.is_calibration_run():
            assert "-C fadc_lg_noise=0.0" in script_content

    run_script = corsika_simtel_runner.runner_service.get_file_name(
        "multi_pipe_script", run_number=1
    )
    assert "|| echo 'Fan-out failed'" not in run_script.read_text(encoding="utf-8")


def test_export_multipipe_script_saves_corsika_output(corsika_config_mock_array_model):
    runner = CorsikaSimtelRunner(
        corsika_config=corsika_config_mock_array_model,
        label="test-corsika-simtel-runner",
        save_corsika_output=True,
    )

    runner._export_multipipe_script(run_number=1)

    script = runner.runner_service.get_file_name("multi_pipe_config", run_number=1)
    corsika_output = runner.corsika_runner.runner_service.get_file_name(
        "corsika_output", run_number=1
    )
    with open(script) as file:
        script_content = file.read()

    assert script_content.count("zstd >") == 1
    assert f"zstd > {corsika_output}" in script_content


def _fake_telescope(name, mirror_class=1):
    telescope = mock.MagicMock()
    telescope.name = name
    telescope.get_parameter_value.side_effect = lambda par: {"mirror_class": mirror_class}[par]
    telescope.get_parameter_value_with_unit.return_value = 12.5 * u.m
    return telescope


@pytest.fixture
def shower_route_runner(corsika_config_mock_array_model, mocker):
    """Runner with two telescope types (LST-like single mirror, one dual-mirror) and a route."""
    mocker.patch.object(CorsikaSimtelRunner, "_shower_route_active", return_value=True)
    mocker.patch.object(CorsikaSimtelRunner, "_minimum_input_plane_height_m", return_value=29.9)
    runner = CorsikaSimtelRunner(
        corsika_config=corsika_config_mock_array_model,
        label="test-corsika-simtel-runner",
    )
    array_model = runner.simulator_array[0].array_model
    array_model.telescope_models = {
        "LSTN-01": _fake_telescope("LST"),
        "LSTN-02": _fake_telescope("LST"),
        "MSTN-01": _fake_telescope("MST", mirror_class=2),
    }
    array_model.site = "South"
    array_model.sim_telarray_seed = _seed_namespace("11,22", instrument_seed=11, simulation_seed=22)
    return runner


def _seed_namespace(seed_string, instrument_seed=None, simulation_seed=None):
    """Stand-in for SimtelSeeds with the attributes used by the shower route."""
    return SimpleNamespace(
        seed_string=seed_string, instrument_seed=instrument_seed, simulation_seed=simulation_seed
    )


@pytest.mark.parametrize(
    ("sim_telarray_seed", "expected"),
    [
        # generated seed pair
        (_seed_namespace("11,22", instrument_seed=11, simulation_seed=22), (11, 22)),
        # fixed test seeds: no instrument seed attribute, both seeds in the seed string
        (_seed_namespace("33,44"), (33, 44)),
        # seed file: instrument seed from the seed object, simulation seed as tracing seed
        (
            _seed_namespace(
                "file-by-run:/tmp/seeds.txt,55", instrument_seed=66, simulation_seed=55
            ),
            (66, 55),
        ),
    ],
)
def test_shower_route_seeds(sim_telarray_seed, expected):
    assert _shower_route_seeds(sim_telarray_seed) == expected


def test_shower_route_seeds_seed_file_without_instrument_seed_raises():
    seeds = _seed_namespace("file-by-run:/tmp/seeds.txt,55", simulation_seed=55)

    with pytest.raises(ValueError, match="Instrument seed is not set"):
        _shower_route_seeds(seeds)


def test_export_multipipe_script_shower_route_seeds_and_work_directories(shower_route_runner):
    runner = shower_route_runner
    runner._export_multipipe_script(run_number=1)

    config = runner.runner_service.get_file_name("multi_pipe_config", run_number=1)
    lines = [line for line in config.read_text(encoding="utf-8").splitlines() if line]
    map_dir = (
        runner.simulator_array[0]
        .runner_service.get_file_name("telescope_route_map", run_number=1)
        .parent
    )
    simulator_runner_service = runner.simulator_array[0].runner_service
    resources = simulator_runner_service.get_file_name(
        file_type="shower_route_resources", run_number=1
    )
    for group in ("LST", "MST"):
        line = next(line for line in lines if f"shower_route_{group}" in line)
        # Instrument seed and tracing seed of the sim_telarray seeds.
        assert "--seed 11 --tracing-seed 22" in line
        # Work directory of the telescope type, next to the telescope map.
        assert f"--output-path {map_dir / f'shower_route_{group}'}" in line
        # The route is wrapped in process accounting with its own resources file.
        assert "--role shower_route" in line
        assert str(resources.with_name(f"{group}_{resources.name}")) in line


def _second_model_version_array(runner, model_version):
    """Copy of the first simulator array with another model version (own runner service)."""
    first = runner.simulator_array[0]
    array_model = copy.copy(first.array_model)
    array_model.model_version = model_version
    second = copy.copy(first)
    second.array_model = array_model
    second.runner_service = runner_services.RunnerServices(
        first.runner_service.config,
        run_type=first.runner_service.run_type,
        array_model=array_model,
    )
    return second


@pytest.mark.parametrize("mode", ["stream", "disk"])
def test_shower_route_files_per_model_version(shower_route_runner, mode):
    runner = shower_route_runner
    runner.shower_route_mode = mode
    first = runner.simulator_array[0]
    second = _second_model_version_array(runner, "7.0.0")
    runner.simulator_array.append(second)

    runner._export_multipipe_script(run_number=1)

    map_first = first.runner_service.get_file_name("telescope_route_map", run_number=1)
    map_second = second.runner_service.get_file_name("telescope_route_map", run_number=1)
    assert map_first != map_second
    assert map_first.exists()
    assert map_second.exists()

    config_text = runner.runner_service.get_file_name("multi_pipe_config", run_number=1).read_text(
        encoding="utf-8"
    )
    script_text = runner.runner_service.get_file_name("multi_pipe_script", run_number=1).read_text(
        encoding="utf-8"
    )
    # Stream mode: routes are in the multipipe config; disk mode: in the multipipe script.
    route_text = config_text + script_text
    assert str(map_first) in route_text
    assert str(map_second) in route_text

    if mode == "disk":
        camera_first = runner._camera_file(first, 1, "LST")
        camera_second = runner._camera_file(second, 1, "LST")
        assert camera_first != camera_second
        assert str(camera_first) in script_text
        assert str(camera_second) in script_text


def test_invalid_shower_route_mode_raises(corsika_config_mock_array_model):
    with pytest.raises(ValueError, match="Invalid shower route mode"):
        CorsikaSimtelRunner(
            corsika_config=corsika_config_mock_array_model, shower_route_mode="file"
        )


def test_export_multipipe_script_shower_route_stream(shower_route_runner):
    runner = shower_route_runner
    runner._export_multipipe_script(run_number=1)

    config = runner.runner_service.get_file_name("multi_pipe_config", run_number=1)
    lines = [line for line in config.read_text(encoding="utf-8").splitlines() if line]
    # One line per telescope type: the route piped into sim_telarray.
    assert len(lines) == 2
    assert all(" | " in line for line in lines)
    assert all("simtools.simtel.shower_route" in line for line in lines)
    assert all("--input -" in line and "--output -" in line for line in lines)
    assert all(line.endswith("-C show=all -") for line in lines)
    single = next(line for line in lines if "--telescope-ids 0,1" in line)
    dual = next(line for line in lines if "--telescope-ids 2" in line)
    assert "-C Bypass_Optics=1" in single
    assert "-C Bypass_Optics=2" in dual
    assert "zstd >" not in config.read_text(encoding="utf-8")

    telescope_map = (
        runner.simulator_array[0]
        .runner_service.get_file_name("telescope_route_map", run_number=1)
        .read_text(encoding="utf-8")
    )
    assert '"0"' in telescope_map
    assert '"dual_mirror": true' in telescope_map
    assert '"input_plane_height_m": 29.9' in telescope_map


def test_export_multipipe_script_shower_route_disk(shower_route_runner):
    runner = shower_route_runner
    runner.shower_route_mode = "disk"
    runner._export_multipipe_script(run_number=1)

    config = runner.runner_service.get_file_name("multi_pipe_config", run_number=1)
    config_text = config.read_text(encoding="utf-8")
    # The CORSIKA IACT file is kept; no route in the pipe.
    assert "zstd > " in config_text
    assert "simtools.simtel.shower_route" not in config_text

    script = runner.runner_service.get_file_name("multi_pipe_script", run_number=1)
    text = script.read_text(encoding="utf-8")
    lines = text.splitlines()
    multipipe_line = next(i for i, line in enumerate(lines) if "multipipe_corsika" in line)
    route_lines = [i for i, line in enumerate(lines) if "simtools.simtel.shower_route" in line]
    bypass_lines = [i for i, line in enumerate(lines) if "Bypass_Optics=" in line]
    assert len(route_lines) == 2
    assert all(i > multipipe_line for i in route_lines)
    assert all("--input " in lines[i] and ".corsika.zst" in lines[i] for i in route_lines)
    assert all(".camera.iact.gz" in lines[i] for i in route_lines)
    # One sim_telarray run per telescope type, after the multipipe, reading the camera file.
    assert len(bypass_lines) == 2
    assert all(i > multipipe_line for i in bypass_lines)
    assert all(".camera.iact.gz" in lines[i] for i in bypass_lines)


def test_export_multipipe_script_without_route_is_unchanged(
    corsika_config_mock_array_model, mocker
):
    mocker.patch.object(CorsikaSimtelRunner, "_shower_route_active", return_value=False)
    runner = CorsikaSimtelRunner(corsika_config=corsika_config_mock_array_model)
    runner._export_multipipe_script(run_number=1)

    config = runner.runner_service.get_file_name("multi_pipe_config", run_number=1)
    text = config.read_text(encoding="utf-8")
    assert "shower_route" not in text
    assert " | " not in text


def test_export_multipipe_script_shower_route_stream_drops_sequential_for_sim_telarray(
    shower_route_runner,
):
    runner = shower_route_runner
    runner._export_multipipe_script(run_number=1)

    config = runner.runner_service.get_file_name("multi_pipe_config", run_number=1)
    lines = [line for line in config.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 2
    for line in lines:
        route_part, simtel_part = line.split(" | ", 1)
        # The route takes part in the sequential protocol, sim_telarray does not.
        assert "MULTIPIPE_SEQUENTIAL" not in route_part
        assert "env -u MULTIPIPE_SEQUENTIAL SIM_TELARRAY_CONFIG_PATH=" in simtel_part
        assert "sim_telarray" in simtel_part


@pytest.mark.parametrize("mode", ["disk", "none"])
def test_export_multipipe_script_without_stream_route_keeps_sequential_env(
    shower_route_runner, mocker, mode
):
    runner = shower_route_runner
    if mode == "disk":
        runner.shower_route_mode = "disk"
    else:
        mocker.patch.object(CorsikaSimtelRunner, "_shower_route_active", return_value=False)
    runner._export_multipipe_script(run_number=1)

    config = runner.runner_service.get_file_name("multi_pipe_config", run_number=1)
    assert "MULTIPIPE_SEQUENTIAL" not in config.read_text(encoding="utf-8")


def test_export_multipipe_script_shower_route_paths_are_absolute(
    shower_route_runner, mocker, monkeypatch, tmp_test_directory
):
    runner = shower_route_runner
    mocker.patch.object(
        settings.config, "_args", {"simulation_models_path": "relative/simulation-models"}
    )
    monkeypatch.chdir(tmp_test_directory)
    runner._export_multipipe_script(run_number=1)

    config = runner.runner_service.get_file_name("multi_pipe_config", run_number=1)
    for line in config.read_text(encoding="utf-8").splitlines():
        if "simtools.simtel.shower_route" not in line:
            continue
        route_part = line.split(" | ", 1)[0]
        args = shlex.split(route_part.split(" 2> ")[0])
        for flag in (
            "--telescope-map",
            "--corsika-card",
            "--output-path",
            "--simulation-models-path",
        ):
            value = args[args.index(flag) + 1]
            assert Path(value).is_absolute(), value
        models_path = args[args.index("--simulation-models-path") + 1]
        assert models_path == str(Path(tmp_test_directory, "relative/simulation-models").resolve())


def test_export_multipipe_script_route_stderr_goes_to_log_file(shower_route_runner):
    runner = shower_route_runner
    runner._export_multipipe_script(run_number=1)

    config = runner.runner_service.get_file_name("multi_pipe_config", run_number=1)
    lines = [line for line in config.read_text(encoding="utf-8").splitlines() if line]
    for line in lines:
        route_part = line.split(" | ", 1)[0]
        # stderr only: the route's stdout is the data stream and is not redirected.
        assert " 2> " in route_part
        log_name = route_part.split(" 2> ", 1)[1].strip().strip("'\"")
        assert log_name.endswith("shower_route.log")
        assert "--log" not in route_part


def test_export_multipipe_script_disk_route_stderr_goes_to_log_file(shower_route_runner):
    runner = shower_route_runner
    runner.shower_route_mode = "disk"
    runner._export_multipipe_script(run_number=1)

    script = runner.runner_service.get_file_name("multi_pipe_script", run_number=1)
    route_lines = [
        line
        for line in script.read_text(encoding="utf-8").splitlines()
        if "simtools.simtel.shower_route" in line
    ]
    assert len(route_lines) == 2
    assert all(" 2> " in line and "shower_route.log" in line for line in route_lines)

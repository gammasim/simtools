#!/usr/bin/python3

import logging
import pathlib
import shutil

import pytest

from simtools.corsika.corsika_runner import CorsikaRunner

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture()
def corsika_runner(corsika_config, io_handler, simtel_path):
    corsika_runner = CorsikaRunner(
        corsika_config=corsika_config,
        simtel_path=simtel_path,
        label="test-corsika-runner",
        use_multipipe=False,
    )
    return corsika_runner


@pytest.fixture()
def corsika_file(io_handler):
    corsika_file = io_handler.get_input_data_file(
        file_name="run1_proton_za20deg_azm0deg_North_1LST_test-lst-array.corsika.zst", test=True
    )
    return corsika_file


def test_corsika_runner(corsika_runner):
    cr = corsika_runner
    assert "corsika_simtel" not in str(cr._directory["output"])
    assert "corsika" in str(cr._directory["output"])
    assert isinstance(cr._directory["data"], pathlib.Path)


def test_load_corsika_data_directories(corsika_runner):
    assert isinstance(corsika_runner._directory, dict)

    for item in corsika_runner._directory.values():
        assert isinstance(item, pathlib.Path)


def test_prepare_run_script(corsika_runner):
    # No run number is given
    script = corsika_runner.prepare_run_script()

    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        assert "/usr/bin/env bash" in script_content
        assert "corsika_autoinputs" in script_content
        assert "sim_telarray/bin/pfp" in script_content

    # Run number is given
    script = corsika_runner.prepare_run_script(run_number=3)

    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        assert "/usr/bin/env bash" in script_content
        assert "corsika_autoinputs" in script_content
        assert "sim_telarray/bin/pfp" in script_content
        assert "-R 3" in script_content


def test_prepare_run_script_with_invalid_run(corsika_runner):
    for run in [-2, "test"]:
        with pytest.raises(ValueError):
            _ = corsika_runner.prepare_run_script(run_number=run)


def test_prepare_run_script_with_extra(corsika_runner, file_has_text):
    extra = ["testing", "testing-extra-2"]
    script = corsika_runner.prepare_run_script(run_number=3, extra_commands=extra)

    assert file_has_text(script, "testing-extra-2")
    with open(script) as f:
        script_content = f.read()
        assert "/usr/bin/env bash" in script_content
        assert "corsika_autoinputs" in script_content
        assert "sim_telarray/bin/pfp" in script_content


def test_prepare_run_script_without_pfp(corsika_runner):
    script = corsika_runner.prepare_run_script(use_pfp=False)

    assert script.exists()
    with open(script) as f:
        script_content = f.read()
        assert "/usr/bin/env bash" in script_content
        assert "corsika_autoinputs" in script_content
        assert "sim_telarray/bin/pfp" not in script_content


def test_get_pfp_command(corsika_runner):
    pfp_command = corsika_runner._get_pfp_command("input_tmp_file")
    assert "sim_telarray/bin/pfp" in pfp_command
    assert "> input_tmp_file" in pfp_command


def test_get_autoinputs_command(corsika_runner):
    autoinputs_command = corsika_runner._get_autoinputs_command(
        run_number=3, input_tmp_file="tmp_file"
    )
    assert "corsika_autoinputs" in autoinputs_command
    assert "-R 3" in autoinputs_command
    assert "--keep-seeds" not in autoinputs_command
    corsika_runner._keep_seeds = True
    autoinputs_command_with_seeds = corsika_runner._get_autoinputs_command(
        run_number=3, input_tmp_file="tmp_file"
    )
    assert "--keep-seeds" in autoinputs_command_with_seeds


def test_get_info_for_file_name(corsika_runner):
    info_for_file_name = corsika_runner.get_info_for_file_name(run_number=1)
    assert info_for_file_name["run"] == 1
    assert info_for_file_name["primary"] == "proton"
    assert info_for_file_name["array_name"] == "test_layout"
    assert info_for_file_name["site"] == "South"
    assert info_for_file_name["label"] == "test-corsika-runner"


def test_get_file_name(corsika_runner, io_handler):
    info_for_file_name = corsika_runner.get_info_for_file_name(run_number=1)
    file_name = "corsika_run000001_proton_South_test_layout_test-corsika-runner"

    assert corsika_runner.get_file_name(
        "corsika_autoinputs_log", **info_for_file_name
    ) == corsika_runner._directory["log"].joinpath(f"log_{file_name}.log.gz")

    assert corsika_runner.get_file_name(
        "corsika_log", **info_for_file_name
    ) == corsika_runner._directory["data"].joinpath(corsika_runner._get_run_directory(1)).joinpath(
        "run1.log"
    )

    script_file_dir = io_handler.get_output_directory("test-corsika-runner", "corsika").joinpath(
        "scripts"
    )
    assert corsika_runner.get_file_name("script", **info_for_file_name) == script_file_dir.joinpath(
        f"{file_name}.sh"
    )

    file_name_for_output = (
        "corsika_run000001_proton_za020deg_azm000deg_South_test_layout_test-corsika-runner.zst"
    )
    assert corsika_runner.get_file_name(
        "output", **info_for_file_name
    ) == corsika_runner._directory["data"].joinpath(corsika_runner._get_run_directory(1)).joinpath(
        file_name_for_output
    )

    sub_log_file_dir = io_handler.get_output_directory("test-corsika-runner", "corsika").joinpath(
        "logs"
    )
    assert corsika_runner.get_file_name(
        "sub_log", **info_for_file_name, mode="out"
    ) == sub_log_file_dir.joinpath(f"log_sub_{file_name}.out")
    with pytest.raises(ValueError):
        corsika_runner.get_file_name("foobar", **info_for_file_name, mode="out")
    assert corsika_runner.get_file_name(
        "sub_log", **info_for_file_name, mode=""
    ) == sub_log_file_dir.joinpath(f"log_sub_{file_name}.log")


def test_has_file(corsika_runner, corsika_file):
    # Copying the corsika file to the expected location and
    # changing its name for the sake of this test.
    # This should not affect the efficiency of this test.
    output_directory = corsika_runner._directory["data"].joinpath(
        corsika_runner._get_run_directory(1)
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        corsika_file,
        output_directory.joinpath(
            "corsika_run000001_proton_za020deg_azm000deg_South_test_layout_test-corsika-runner.zst"
        ),
    )
    assert corsika_runner.has_file(file_type="output", run_number=1)
    assert not corsika_runner.has_file(file_type="corsika_autoinputs_log", run_number=1234)


def test_get_resources(corsika_runner, caplog):
    sub_log_file = corsika_runner.get_file_name(
        file_type="sub_log", **corsika_runner.get_info_for_file_name(None), mode="out"
    )
    with open(sub_log_file, "w", encoding="utf-8") as file:
        lines_to_write = ["RUNTIME 500\n"]
        file.writelines(lines_to_write)
    resources = corsika_runner.get_resources()
    assert isinstance(resources, dict)
    assert "runtime" in resources
    assert resources["runtime"] == 500

    with open(sub_log_file, "w", encoding="utf-8") as file:
        lines_to_write = ["SOMETHING ELSE 500\n"]
        file.writelines(lines_to_write)

    with caplog.at_level(logging.DEBUG):
        resources = corsika_runner.get_resources()
    assert resources["runtime"] is None
    assert "RUNTIME was not found in run log file" in caplog.text


def test_get_run_directory(corsika_runner):
    run_directory = corsika_runner._get_run_directory(1)
    assert run_directory == "run000001"
    run_directory = corsika_runner._get_run_directory(123456)
    assert run_directory == "run123456"
    with pytest.raises(ValueError, match=r"^Run number cannot have more than 6 digits"):
        run_directory = corsika_runner._get_run_directory(1234567)


def test_validate_run_number(corsika_runner):
    assert corsika_runner._validate_run_number(1)
    assert corsika_runner._validate_run_number(123456)
    with pytest.raises(ValueError, match=r"^could not convert string to float"):
        corsika_runner._validate_run_number("test")
    with pytest.raises(ValueError, match=r"^Invalid type of run number"):
        corsika_runner._validate_run_number(1.5)
    with pytest.raises(ValueError, match=r"^Invalid type of run number"):
        corsika_runner._validate_run_number(-1)
    with pytest.raises(ValueError, match=r"^Invalid type of run number"):
        corsika_runner._validate_run_number(123456789)

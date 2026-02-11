import gzip
import logging
from pathlib import Path

import pytest

from simtools.constants import MODEL_PARAMETER_SCHEMA_PATH
from simtools.testing import assertions

logging.getLogger().setLevel(logging.DEBUG)


@pytest.fixture
def test_json_file():
    return Path("tests/resources/reference_point_altitude.json")


@pytest.fixture
def test_yaml_file():
    return MODEL_PARAMETER_SCHEMA_PATH / "num_gains.schema.yml"


@pytest.fixture
def tar_with_log(tmp_test_directory, safe_tar_open):
    """Helper fixture to create tar.gz with log content."""

    def _create_tar(log_content: bytes, filename: str = "test_logs.tar.gz"):
        tmp_path = Path(tmp_test_directory)
        tar_path = tmp_path / filename
        with safe_tar_open(tar_path, "w:gz") as tar:
            log_gz = tmp_path / "test.log.gz"
            with gzip.open(log_gz, "wb") as gz:
                gz.write(log_content)
            tar.add(log_gz, arcname="test.log.gz")
        return tar_path

    return _create_tar


def test_assert_file_type_json(test_json_file, test_yaml_file):
    assert assertions.assert_file_type("json", test_json_file)
    assert not assertions.assert_file_type("json", "tests/resources/does_not_exist.json")
    assert not assertions.assert_file_type("json", test_yaml_file)

    assert assertions.assert_file_type("json", Path(test_json_file))


def test_assert_file_type_yaml(test_json_file, test_yaml_file, caplog):
    assert assertions.assert_file_type("yaml", test_yaml_file)
    assert assertions.assert_file_type("yml", test_yaml_file)
    assert not assertions.assert_file_type("yml", "tests/resources/does_not_exit.schema.yml")

    assert not assertions.assert_file_type(
        "yaml", "tests/resources/telescope_positions-South-ground.ecsv"
    )


def test_assert_file_type_others(caplog):
    with caplog.at_level(logging.INFO):
        assert assertions.assert_file_type(
            "ecsv", "tests/resources/telescope_positions-South-ground.ecsv"
        )
    assert (
        "File type test is checking suffix only for tests/resources/"
        "telescope_positions-South-ground.ecsv (suffix: ecsv)" in caplog.text
    )


def test_assert_no_suffix():
    assert not assertions.assert_file_type("yml", "tests/resources/does_not_exit_yml")


@pytest.mark.parametrize(
    ("filename", "file_content", "mock_function", "expected_result"),
    [
        ("test.log", "test log content\n", "check_plain_logs", True),
        ("test.log", "test log content\n", "check_plain_logs", False),
        ("test_logs.tar.gz", b"fake tar content", "check_tar_logs", True),
        ("test_logs.tar.gz", b"fake tar content", "check_tar_logs", False),
    ],
)
def test_check_log_files(tmp_path, mocker, filename, file_content, mock_function, expected_result):
    log_file = tmp_path / filename
    if isinstance(file_content, bytes):
        log_file.write_bytes(file_content)
    else:
        log_file.write_text(file_content)

    mock_check = mocker.patch(f"simtools.testing.assertions.{mock_function}")
    mock_check.return_value = expected_result

    file_test = (
        {"wanted_patterns": ["test"]} if expected_result else {"forbidden_patterns": ["error"]}
    )
    result = assertions.check_log_files(log_file, file_test)

    assert result is expected_result
    mock_check.assert_called_once_with(log_file, file_test)


def test_check_output_from_sim_telarray_no_expected_output(mocker):
    mock_file = Path("test_file.zst")
    file_test = {"some_other_key": "value"}

    result = assertions.check_output_from_sim_telarray(mock_file, file_test)

    assert result is True


@pytest.mark.parametrize(
    ("file_test_config", "expected_result", "check_showers", "event_type"),
    [
        (
            {"expected_sim_telarray_output": {"event_type": "shower", "some_key": "some_value"}},
            True,
            True,
            "shower",
        ),
        ({"expected_sim_telarray_metadata": {"version": "1.0"}}, True, True, "shower"),
        ({"expected_sim_telarray_output": {"event_type": "background"}}, True, False, "background"),
        ({"expected_sim_telarray_output": {"event_type": "shower"}}, False, True, "shower"),
    ],
)
def test_check_output_from_sim_telarray(
    mocker, file_test_config, expected_result, check_showers, event_type
):
    mock_file = Path("test_file.zst")

    mock_assert_output = mocker.patch(
        "simtools.simtel.simtel_output_validator.assert_expected_sim_telarray_output",
        return_value=expected_result,
    )
    mock_assert_metadata = mocker.patch(
        "simtools.simtel.simtel_output_validator.assert_expected_sim_telarray_metadata",
        return_value=True,
    )
    mock_assert_events = mocker.patch(
        "simtools.simtel.simtel_output_validator.assert_events_of_type", return_value=True
    )
    mock_assert_showers = mocker.patch(
        "simtools.simtel.simtel_output_validator.assert_n_showers_and_energy_range",
        return_value=True,
    )

    result = assertions.check_output_from_sim_telarray(mock_file, file_test_config)

    assert result is expected_result

    if "expected_sim_telarray_output" in file_test_config:
        mock_assert_output.assert_called_once_with(
            file=mock_file,
            expected_sim_telarray_output=file_test_config["expected_sim_telarray_output"],
        )
        mock_assert_events.assert_called_once_with(mock_file, event_type=event_type)
        if check_showers:
            mock_assert_showers.assert_called_once_with(mock_file)
        else:
            mock_assert_showers.assert_not_called()

    if "expected_sim_telarray_metadata" in file_test_config:
        mock_assert_metadata.assert_called_once_with(
            file=mock_file,
            expected_sim_telarray_metadata=file_test_config["expected_sim_telarray_metadata"],
        )

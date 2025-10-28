#!/usr/bin/python3

import json
import logging
from pathlib import Path

import pytest
import yaml
from astropy.table import Table

from simtools.testing import validate_output

logging.getLogger().setLevel(logging.DEBUG)

PATH_TO_OUTPUT = "/path/to/output"
PATCH_TO_VALIDATE_CFG = "simtools.testing.validate_output._validate_simtel_cfg_files"
PATH_CFG_6 = "/path/to/simtel_cfg_6.0.0.cfg"
PATH_CFG_7 = "/path/to/simtel_cfg_7.0.0.cfg"
TEST_PARAM_JSON = "test_param.json"


@pytest.fixture
def create_json_file(tmp_test_directory):
    def _create_json_file(file_name, content):
        _file = tmp_test_directory / file_name
        _file.write_text(json.dumps(content), encoding="utf-8")
        return _file

    return _create_json_file


@pytest.fixture
def create_yaml_file(tmp_test_directory):
    def _create_yaml_file(file_name, content):
        _file = tmp_test_directory / file_name
        with open(_file, "w", encoding="utf-8") as f:
            yaml.dump(content, f)
        return _file

    return _create_yaml_file


@pytest.fixture
def create_ecsv_file(tmp_test_directory):
    def _create_ecsv_file(file_name, content):
        table = Table(content)
        file_path = tmp_test_directory / file_name
        table.write(file_path, format="ascii.ecsv")
        return file_path

    return _create_ecsv_file


@pytest.fixture
def file_name():
    def _file_name(counter, suffix):
        return f"file{counter}.{suffix}"

    return _file_name


@pytest.fixture
def test_path():
    return "/path/to/reference/file"


@pytest.fixture
def output_path():
    return PATH_TO_OUTPUT


@pytest.fixture
def mock_validate_application_output(mocker):
    return mocker.patch("simtools.testing.validate_output.validate_application_output")


@pytest.fixture
def mock_path_exists(mocker):
    return mocker.patch("simtools.testing.validate_output.Path.exists", return_value=True)


@pytest.fixture
def mock_check_output(mocker):
    return mocker.patch("simtools.testing.assertions.check_output_from_sim_telarray")


@pytest.fixture
def mock_validate_reference_output_file(mocker):
    return mocker.patch("simtools.testing.validate_output._validate_reference_output_file")


@pytest.fixture
def mock_validate_simtel_cfg_files(mocker):
    return mocker.patch(PATCH_TO_VALIDATE_CFG)


@pytest.fixture
def mock_validate_model_parameter_json_file(mocker):
    return mocker.patch("simtools.testing.validate_output._validate_model_parameter_json_file")


@pytest.fixture
def mock_validate_output_path_and_file(mocker):
    return mocker.patch("simtools.testing.validate_output._validate_output_path_and_file")


@pytest.fixture
def mock_assert_file_type(mocker):
    return mocker.patch("simtools.testing.assertions.assert_file_type")


def test_compare_json_files_float_strings(create_json_file, file_name):
    content = {"key": 1, "value": "1.23 4.56 7.89", "schema_version": "1.0.0"}
    file1 = create_json_file(file_name(1, "json"), content)
    file2 = create_json_file(file_name(2, "json"), content)

    assert validate_output.compare_json_or_yaml_files(file1, file2)

    content3 = {"key": 2, "value": "1.23 4.56 7.80", "schema_version": "2.0.0"}
    file3 = create_json_file(file_name(3, "json"), content3)
    assert not validate_output.compare_json_or_yaml_files(file1, file3)


def test_compare_json_files_equal_dicts(create_json_file, file_name):
    content1 = {"key": 1, "value": 5}
    file1 = create_json_file(file_name(1, "json"), content1)
    content2 = {"key": 1, "value": 5, "extra": "extra"}
    file2 = create_json_file(file_name(2, "json"), content2)
    assert not validate_output.compare_json_or_yaml_files(file1, file2)

    content3 = {"different_key": 1, "value": 5}
    file3 = create_json_file(file_name(2, "json"), content3)
    assert not validate_output.compare_json_or_yaml_files(file1, file3)


def test_compare_json_files_equal_integers(create_json_file, file_name):
    content = {"key": 1, "value": 5}
    file1 = create_json_file(file_name(1, "json"), content)
    file2 = create_json_file(file_name(2, "json"), content)

    assert validate_output.compare_json_or_yaml_files(file1, file2)

    content3 = {"key": 2, "value": 7}
    file3 = create_json_file(file_name(3, "json"), content3)
    assert not validate_output.compare_json_or_yaml_files(file1, file3)


def test_compare_json_files_equal_floats(create_json_file, file_name):
    content = {"key": 1, "value": 5.5}
    file1 = create_json_file(file_name(1, "json"), content)
    file2 = create_json_file(file_name(2, "json"), content)

    assert validate_output.compare_json_or_yaml_files(file1, file2)

    content3 = {"key": 1, "value": 5.75}
    file3 = create_json_file(file_name(3, "json"), content3)
    assert not validate_output.compare_json_or_yaml_files(file1, file3)

    assert validate_output.compare_json_or_yaml_files(file1, file3, tolerance=0.5)


def test_compare_json_files_list_of_floats(create_json_file, file_name):
    content = {"key": 1, "value": [5.5, 10.5]}
    file1 = create_json_file(file_name(1, "json"), content)
    file2 = create_json_file(file_name(2, "json"), content)

    assert validate_output.compare_json_or_yaml_files(file1, file2)

    content3 = {"key": 1, "value": 5.75}
    file3 = create_json_file(file_name(3, "json"), content3)
    assert not validate_output.compare_json_or_yaml_files(file1, file3)

    content4 = {"key": 1, "value": [5.75, 10.75]}
    file4 = create_json_file(file_name(3, "json"), content4)
    assert validate_output.compare_json_or_yaml_files(file1, file4, tolerance=0.5)


def test_compare_yaml_files_float_strings(create_yaml_file, file_name):
    content = {"key": 1, "value": "1.23 4.56 7.89"}
    file1 = create_yaml_file(file_name(1, "yaml"), content)
    file2 = create_yaml_file(file_name(2, "yaml"), content)

    assert validate_output.compare_json_or_yaml_files(file1, file2)

    content3 = {"key": 1, "value": "1.23 4.56 7.80"}
    file3 = create_yaml_file(file_name(3, "yaml"), content3)
    assert not validate_output.compare_json_or_yaml_files(file1, file3)

    assert validate_output.compare_json_or_yaml_files(file1, file3, tolerance=0.5)


def test_compare_yaml_files_equal_integers(create_yaml_file, file_name):
    content = {"key": 1, "value": 5}
    file1 = create_yaml_file(file_name(1, "yaml"), content)
    file2 = create_yaml_file(file_name(2, "yaml"), content)

    assert validate_output.compare_json_or_yaml_files(file1, file2)

    content3 = {"key": 2, "value": 7}
    file3 = create_yaml_file(file_name(3, "yaml"), content3)
    assert not validate_output.compare_json_or_yaml_files(file1, file3)


def test_compare_ecsv_files_equal(create_ecsv_file, file_name):
    content = {"col1": [1.1, 2.2, 3.3], "col2": [4.4, 5.5, 6.6]}
    file1 = create_ecsv_file(file_name(1, "ecsv"), content)
    file2 = create_ecsv_file(file_name(2, "ecsv"), content)

    assert validate_output.compare_ecsv_files(file1, file2)


def test_compare_ecsv_files_different_lengths(create_ecsv_file, file_name):
    content1 = {"col1": [1.1, 2.2, 3.3], "col2": [4.4, 5.5, 6.6]}
    content2 = {"col1": [1.1, 2.2], "col2": [4.4, 5.5]}
    file1 = create_ecsv_file(file_name(1, "yaml"), content1)
    file2 = create_ecsv_file(file_name(2, "ecsv"), content2)

    assert not validate_output.compare_ecsv_files(file1, file2)


def test_compare_ecsv_files_close_values(create_ecsv_file, file_name):
    content1 = {"col1": [1.1001, 2.2001, 3.3001], "col2": [4.4001, 5.5001, 6.6001]}
    content2 = {"col1": [1.1, 2.2, 3.3], "col2": [4.4, 5.5, 6.6]}
    file1 = create_ecsv_file(file_name(1, "ecsv"), content1)
    file2 = create_ecsv_file(file_name(2, "ecsv"), content2)

    assert validate_output.compare_ecsv_files(file1, file2, tolerance=1.0e-3)


def test_compare_ecsv_files_large_difference(create_ecsv_file, file_name):
    content1 = {"col1": [1.1, 2.2, 3.3], "col2": [4.4, 5.5, 6.6]}
    content2 = {"col1": [10.1, 20.2, 30.3], "col2": [40.4, 50.5, 60.6]}
    file1 = create_ecsv_file(file_name(1, "ecsv"), content1)
    file2 = create_ecsv_file(file_name(2, "ecsv"), content2)

    assert not validate_output.compare_ecsv_files(file1, file2)


def test_compare_files_ecsv(create_ecsv_file, file_name):
    content = {"col1": [1.1, 2.2, 3.3], "col2": [4.4, 5.5, 6.6]}
    file1 = create_ecsv_file(file_name(1, "ecsv"), content)
    file2 = create_ecsv_file(file_name(2, "ecsv"), content)

    assert validate_output.compare_files(file1, file2)


def test_compare_files_ecsv_columns(create_ecsv_file, file_name):
    content1 = {"col1": [1.1, 2.2, 3.3], "col2": [4.4, 5.5, 6.6]}
    content2 = {"col1": [1.1, 2.21, 3.31], "col2": [4.47, 5.5, 6.6], "col3": [7.7, 8.8, 9.9]}
    file1 = create_ecsv_file(file_name(1, "ecsv"), content1)
    file2 = create_ecsv_file(file_name(2, "ecsv"), content2)

    assert validate_output.compare_files(file1, file2, 0.5)
    assert validate_output.compare_files(file1, file2, 0.005, [{"test_column_name": "col1"}])
    assert not validate_output.compare_files(file1, file2, 0.005, None)
    assert not validate_output.compare_files(file1, file2, 0.005, [{"test_column_name": "col2"}])
    assert validate_output.compare_files(
        file1,
        file2,
        0.005,
        [{"test_column_name": "col1", "cut_column_name": "col2", "cut_condition": "> 4.5"}],
    )
    # select first column only (same values)
    assert validate_output.compare_files(
        file1,
        file2,
        1.0e-3,
        [{"test_column_name": "col1", "cut_column_name": "col2", "cut_condition": "<5."}],
    )
    # select 2nd/3rd column with larger difference between values
    assert not validate_output.compare_files(
        file1,
        file2,
        1.0e-3,
        [{"test_column_name": "col1", "cut_column_name": "col2", "cut_condition": ">5."}],
    )


def test_compare_files_json(create_json_file, file_name):
    content = {"key": 1, "value": "1.23 4.56 7.89"}
    file1 = create_json_file(file_name(1, "json"), content)
    file2 = create_json_file(file_name(2, "json"), content)

    assert validate_output.compare_files(file1, file2)


def test_compare_files_yaml(create_yaml_file, file_name):
    content = {"key": 1, "value": "1.23 4.56 7.89"}
    file1 = create_yaml_file(file_name(1, "yaml"), content)
    file2 = create_yaml_file(file_name(2, "yaml"), content)

    assert validate_output.compare_files(file1, file2)


def test_compare_files_different_suffixes(create_json_file, create_yaml_file, file_name):
    content = {"key": 1, "value": "1.23 4.56 7.89"}
    file1 = create_json_file(file_name(1, "json"), content)
    file2 = create_yaml_file(file_name(2, "yaml"), content)

    with pytest.raises(ValueError, match="File suffixes do not match"):
        validate_output.compare_files(file1, file2)


def test_compare_files_unknown_type(tmp_test_directory, file_name):
    file1 = tmp_test_directory / file_name(1, "txt")
    file2 = tmp_test_directory / file_name(2, "txt")
    file1.write_text("dummy content", encoding="utf-8")
    file2.write_text("dummy content", encoding="utf-8")

    assert not validate_output.compare_files(file1, file2)


def test_validate_reference_output_file(mocker, output_path, test_path):
    config = {"configuration": {"output_path": output_path, "output_file": "output_file"}}
    integration_test = {
        "reference_output_file": test_path,
        "tolerance": 1.0e-5,
        "test_columns": None,
    }

    mock_compare_files = mocker.patch(
        "simtools.testing.validate_output.compare_files", return_value=True
    )

    validate_output._validate_reference_output_file(config, integration_test)

    mock_compare_files.assert_called_once_with(
        integration_test["reference_output_file"],
        Path(config["configuration"]["output_path"]).joinpath(
            config["configuration"]["output_file"]
        ),
        integration_test.get("tolerance", 1.0e-5),
        integration_test.get("test_columns", None),
    )


def test_validate_output_path_and_file(output_path, mock_path_exists, mock_check_output):
    config = {
        "configuration": {"output_path": output_path, "data_directory": "/path/to/data"},
        "integration_tests": [{"expected_output": "expected_output"}],
    }
    integration_test = [
        {"path_descriptor": "data_directory", "file": "output_file", "expected_output": {}}
    ]

    validate_output._validate_output_path_and_file(config, integration_test)

    mock_path_exists.assert_called()
    mock_check_output.assert_called_once_with(
        Path(config["configuration"]["data_directory"]).joinpath(integration_test[0]["file"]),
        {"path_descriptor": "data_directory", "file": "output_file", "expected_output": {}},
    )

    wrong_integration_test = [
        {"path_descriptor": "wrong_path", "file": "output_file", "expected_output": {}}
    ]
    with pytest.raises(
        KeyError, match=r"Path wrong_path not found in integration test configuration."
    ):
        validate_output._validate_output_path_and_file(config, wrong_integration_test)


def test_validate_application_output_no_integration_tests(mocker, output_path):
    config = {"configuration": {"output_path": output_path}}
    mock_logger_info = mocker.patch("simtools.testing.validate_output._logger.info")

    validate_output.validate_application_output(config)

    mock_logger_info.assert_not_called()


def test_validate_application_output_with_reference_output_file(
    output_path,
    test_path,
    mock_assert_file_type,
    mock_validate_output_path_and_file,
    mock_validate_reference_output_file,
    mock_validate_simtel_cfg_files,
    mock_validate_model_parameter_json_file,
):
    config = {
        "configuration": {"output_path": output_path},
        "integration_tests": [
            {"reference_output_file": test_path},
            {
                "test_simtel_cfg_files": {"6.0.0": test_path},
                "model_parameter_validation": {
                    "reference_parameter_name": "test_param",
                    "parameter_file": "test_param-0.0.99.json",
                    "tolerance": 1.0e-5,
                },
            },
        ],
    }

    validate_output.validate_application_output(config)

    mock_validate_reference_output_file.assert_called_once_with(
        config, config["integration_tests"][0]
    )
    mock_validate_output_path_and_file.assert_not_called()
    mock_assert_file_type.assert_not_called()
    mock_validate_simtel_cfg_files.assert_not_called()

    validate_output.validate_application_output(config, "6.0.0")
    mock_validate_simtel_cfg_files.assert_called_once()
    mock_validate_model_parameter_json_file.assert_called_once()


def test_validate_application_output_with_assertion_error(output_path):
    test_path = "not_there"
    config = {
        "configuration": {"output_path": output_path},
        "test_output_files": [{"path_descriptor": "output_path", "file": test_path}],
    }
    with pytest.raises(
        AssertionError, match=r"Output file /path/to/output/not_there does not exist."
    ):
        validate_output._validate_output_path_and_file(
            config, [{"path_descriptor": "output_path", "file": test_path}]
        )


def test_validate_application_output_with_file_type(
    output_path,
    mock_assert_file_type,
    mock_validate_output_path_and_file,
    mock_validate_reference_output_file,
):
    config = {
        "configuration": {"output_path": output_path, "output_file": "output_file"},
        "integration_tests": [
            {"file_type": "ecsv", "test_output_files": [], "output_file": "output_file"}
        ],
    }

    validate_output.validate_application_output(config)

    mock_validate_reference_output_file.assert_not_called()
    mock_validate_output_path_and_file.assert_called()
    assert mock_validate_output_path_and_file.call_count == 2
    mock_assert_file_type.assert_called_once_with(
        "ecsv",
        Path(config["configuration"]["output_path"]).joinpath(
            config["configuration"]["output_file"]
        ),
    )


def test_compare_simtel_cfg_files(tmp_test_directory):
    file1 = Path("tests/resources/sim_telarray_configurations/CTA-North-LSTN-01-6.0_test.cfg")
    file2 = Path("tests/resources/sim_telarray_configurations/CTA-North-LSTN-01-6.0_test.cfg")

    assert validate_output._compare_simtel_cfg_files(file1, file2)

    with open(file1) as f1:
        lines1 = f1.readlines()

    # additional line in file
    file3 = tmp_test_directory / "file3.cfg"
    with open(file3, "a") as f3:
        f3.write("".join(lines1))
        f3.write("Additional line\n")
    assert not validate_output._compare_simtel_cfg_files(file1, file3)

    # change of values
    file4 = tmp_test_directory / "file4.cfg"
    with open(file4, "a") as f3:
        f3.write("".join(lines1).replace("1", "2"))
    assert not validate_output._compare_simtel_cfg_files(file1, file4)


def test_validate_simtel_cfg_files(mocker, test_path):
    mocker.patch("simtools.testing.validate_output._compare_simtel_cfg_files", return_value=True)
    config = {
        "configuration": {
            "output_path": PATH_TO_OUTPUT,
            "model_version": "3.4.5",
            "label": "label",
        },
        "integration_tests": [{"test_simtel_cfg_files": test_path}],
    }
    validate_output._validate_simtel_cfg_files(config, test_path)


def test_compare_value_from_parameter_dict():
    data_1 = "mirror_list.dat"
    data_2 = "mirror_list.dat"
    data_3 = "pixel_list.dat"
    assert validate_output._compare_value_from_parameter_dict(data_1, data_2)
    assert not validate_output._compare_value_from_parameter_dict(data_1, data_3)


def test_test_simtel_cfg_files_with_command_line_version(mocker):
    mock_validate_simtel_cfg_files = mocker.patch(PATCH_TO_VALIDATE_CFG)
    config = {"configuration": {"output_path": PATH_TO_OUTPUT}}
    integration_test = {
        "test_simtel_cfg_files": {
            "6.0.0": PATH_CFG_6,
            "7.0.0": PATH_CFG_7,
        }
    }
    from_command_line = ["5.0.0", "6.0.0"]
    from_config_file = None

    validate_output._test_simtel_cfg_files(
        config, integration_test, from_command_line, from_config_file
    )

    mock_validate_simtel_cfg_files.assert_called_once_with(config, PATH_CFG_6)


def test_test_simtel_cfg_files_with_config_file_version(mocker):
    mock_validate_simtel_cfg_files = mocker.patch(PATCH_TO_VALIDATE_CFG)
    config = {"configuration": {"output_path": PATH_TO_OUTPUT}}
    integration_test = {
        "test_simtel_cfg_files": {
            "6.0.0": PATH_CFG_6,
            "7.0.0": PATH_CFG_7,
        }
    }
    from_command_line = None
    from_config_file = ["7.0.0", "8.0.0"]

    validate_output._test_simtel_cfg_files(
        config, integration_test, from_command_line, from_config_file
    )

    mock_validate_simtel_cfg_files.assert_called_once_with(config, PATH_CFG_7)


def test_test_simtel_cfg_files_with_single_version(mocker):
    mock_validate_simtel_cfg_files = mocker.patch(PATCH_TO_VALIDATE_CFG)
    config = {"configuration": {"output_path": PATH_TO_OUTPUT}}
    integration_test = {
        "test_simtel_cfg_files": {
            "6.0.0": PATH_CFG_6,
            "7.0.0": PATH_CFG_7,
        }
    }
    from_command_line = "6.0.0"
    from_config_file = None

    validate_output._test_simtel_cfg_files(
        config, integration_test, from_command_line, from_config_file
    )

    mock_validate_simtel_cfg_files.assert_called_once_with(config, PATH_CFG_6)


def test_test_simtel_cfg_files_no_matching_version(mocker):
    mock_validate_simtel_cfg_files = mocker.patch(PATCH_TO_VALIDATE_CFG)
    config = {"configuration": {"output_path": PATH_TO_OUTPUT}}
    integration_test = {
        "test_simtel_cfg_files": {
            "6.0.0": PATH_CFG_6,
            "7.0.0": PATH_CFG_7,
        }
    }
    from_command_line = ["5.0.0"]
    from_config_file = ["8.0.0"]

    validate_output._test_simtel_cfg_files(
        config, integration_test, from_command_line, from_config_file
    )

    mock_validate_simtel_cfg_files.assert_not_called()


def test_test_simtel_cfg_files_no_test_simtel_cfg_files(mocker):
    mock_validate_simtel_cfg_files = mocker.patch(PATCH_TO_VALIDATE_CFG)
    config = {"configuration": {"output_path": PATH_TO_OUTPUT}}
    integration_test = {}

    validate_output._test_simtel_cfg_files(config, integration_test, None, None)

    mock_validate_simtel_cfg_files.assert_not_called()


def test_validate_model_parameter_json_file(mocker, output_path):
    mock_db_handler = mocker.patch("simtools.db.db_handler.DatabaseHandler")
    mock_collect_data_from_file = mocker.patch("simtools.io.ascii_handler.collect_data_from_file")
    mock_compare_value = mocker.patch(
        "simtools.testing.validate_output._compare_value_from_parameter_dict"
    )

    mock_db_instance = mock_db_handler.return_value
    mock_db_instance.get_model_parameter.return_value = {
        "reference_param": {"value": [1.0, 2.0, 3.0]}
    }
    mock_collect_data_from_file.return_value = {"value": [1.0, 2.0, 3.0]}

    config = {
        "configuration": {
            "output_path": output_path,
            "telescope": "test_telescope",
            "model_version": "1.0.0",
            "site": "test_site",
        }
    }
    model_parameter_validation = {
        "reference_parameter_name": "reference_param",
        "parameter_file": TEST_PARAM_JSON,
        "tolerance": 1.0e-5,
    }

    validate_output._validate_model_parameter_json_file(
        config, model_parameter_validation, db_config=None
    )

    mock_db_handler.assert_called_once_with(db_config=None)
    mock_db_instance.get_model_parameter.assert_called_once_with(
        parameter="reference_param",
        site="test_site",
        array_element_name="test_telescope",
        model_version="1.0.0",
    )
    mock_collect_data_from_file.assert_called_once_with(
        Path(output_path) / "test_telescope" / TEST_PARAM_JSON
    )
    mock_compare_value.assert_called_once_with([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1.0e-5)


def test_validate_model_parameter_json_file_mismatch(mocker, output_path):
    mock_db_handler = mocker.patch("simtools.db.db_handler.DatabaseHandler")
    mock_collect_data_from_file = mocker.patch("simtools.io.ascii_handler.collect_data_from_file")
    mock_compare_value = mocker.patch(
        "simtools.testing.validate_output._compare_value_from_parameter_dict", return_value=False
    )

    mock_db_instance = mock_db_handler.return_value
    mock_db_instance.get_model_parameter.return_value = {
        "reference_param": {"value": [1.0, 2.0, 3.0]}
    }
    mock_collect_data_from_file.return_value = {"value": [1.1, 2.1, 3.1]}

    config = {
        "configuration": {
            "output_path": output_path,
            "telescope": "test_telescope",
            "model_version": "1.0.0",
            "site": "test_site",
        }
    }
    model_parameter_validation = {
        "reference_parameter_name": "reference_param",
        "parameter_file": TEST_PARAM_JSON,
        "tolerance": 1.0e-5,
    }

    with pytest.raises(AssertionError):
        validate_output._validate_model_parameter_json_file(
            config, model_parameter_validation, db_config=None
        )

    mock_db_handler.assert_called_once_with(db_config=None)
    mock_db_instance.get_model_parameter.assert_called_once_with(
        parameter="reference_param",
        site="test_site",
        array_element_name="test_telescope",
        model_version="1.0.0",
    )
    mock_collect_data_from_file.assert_called_once_with(
        Path(output_path) / "test_telescope" / TEST_PARAM_JSON
    )
    mock_compare_value.assert_called_once_with([1.1, 2.1, 3.1], [1.0, 2.0, 3.0], 1.0e-5)


def test_resolve_output_file_path_exact_match(tmp_test_directory):
    tmp_dir = Path(str(tmp_test_directory))
    output_path = tmp_dir / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    version_dir = output_path / "6.0"
    version_dir.mkdir(parents=True, exist_ok=True)
    test_file = version_dir / "test.md"
    test_file.write_text("test content")

    resolved = validate_output._resolve_output_file_path(output_path, "6.0/test.md", "6.0")
    assert resolved == test_file


def test_resolve_output_file_path_glob_fallback(tmp_test_directory):
    tmp_dir = Path(str(tmp_test_directory))
    output_path = tmp_dir / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    version_dir = output_path / "6.0.2"
    version_dir.mkdir(parents=True, exist_ok=True)
    test_file = version_dir / "test.md"
    test_file.write_text("test content")

    resolved = validate_output._resolve_output_file_path(output_path, "6.0/test.md", "6.0")
    assert resolved == test_file


def test_resolve_output_file_path_multiple_versions(tmp_test_directory):
    tmp_dir = Path(str(tmp_test_directory))
    output_path = tmp_dir / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    version_dir1 = output_path / "6.0.1"
    version_dir1.mkdir(parents=True, exist_ok=True)
    version_dir2 = output_path / "6.0.2"
    version_dir2.mkdir(parents=True, exist_ok=True)
    test_file1 = version_dir1 / "test.md"
    test_file1.write_text("test content 1")
    test_file2 = version_dir2 / "test.md"
    test_file2.write_text("test content 2")

    resolved = validate_output._resolve_output_file_path(output_path, "6.0/test.md", "6.0")
    # Should resolve to highest (latest) patch version
    assert resolved == test_file2


def test_resolve_output_file_path_no_match(tmp_test_directory):
    tmp_dir = Path(str(tmp_test_directory))
    output_path = tmp_dir / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    resolved = validate_output._resolve_output_file_path(output_path, "6.0/test.md", "6.0")
    assert resolved == output_path / "6.0" / "test.md"


def test_resolve_output_file_path_version_in_filename(tmp_test_directory):
    tmp_dir = Path(str(tmp_test_directory))
    output_path = tmp_dir / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    actual_filename = "gamma_run000001_za40deg_azm180deg_North_alpha_6.0.2_check_output.log"
    test_file = output_path / actual_filename
    test_file.write_text("test content")

    expected_filename = "gamma_run000001_za40deg_azm180deg_North_alpha_6.0_check_output.log"
    resolved = validate_output._resolve_output_file_path(output_path, expected_filename, "6.0")
    assert resolved == test_file


def test_resolve_output_file_path_version_in_filename_exact_match(tmp_test_directory):
    tmp_dir = Path(str(tmp_test_directory))
    output_path = tmp_dir / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    test_file = output_path / "file_6.0_name.txt"
    test_file.write_text("test content")

    resolved = validate_output._resolve_output_file_path(output_path, "file_6.0_name.txt", "6.0")
    assert resolved == test_file


def test_resolve_output_file_path_version_in_filename_no_match(tmp_test_directory):
    tmp_dir = Path(str(tmp_test_directory))
    output_path = tmp_dir / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    resolved = validate_output._resolve_output_file_path(output_path, "file_6.0_name.txt", "6.0")
    assert resolved == output_path / "file_6.0_name.txt"


def test_lines_match_with_version_flexibility():
    ref = "ModelVersion 6.0"
    test = "ModelVersion 6.0.2"
    assert validate_output._lines_match_with_version_flexibility(ref, test)

    ref = "include CTA-North-LSTN-01_6.0_test.cfg"
    test = "include CTA-North-LSTN-01_6.0.2_test.cfg"
    assert validate_output._lines_match_with_version_flexibility(ref, test)

    ref = "config_version 6.0"
    test = "config_version 6.0.2"
    assert validate_output._lines_match_with_version_flexibility(ref, test)

    ref = "value 5.0"
    test = "value 5.5"
    assert not validate_output._lines_match_with_version_flexibility(ref, test)


def test_validate_output_path_and_file_without_model_version(
    output_path, mock_path_exists, mock_check_output
):
    config = {
        "configuration": {"output_path": output_path, "data_directory": "/path/to/data"},
        "integration_tests": [{"expected_output": "expected_output"}],
    }
    integration_test = [
        {"path_descriptor": "data_directory", "file": "output_file", "expected_output": {}}
    ]

    validate_output._validate_output_path_and_file(config, integration_test)

    mock_path_exists.assert_called()
    mock_check_output.assert_called_once()


def test_validate_output_path_and_file_without_model_version_real_file(
    tmp_test_directory, mock_check_output
):
    tmp_dir = Path(str(tmp_test_directory))
    data_dir = tmp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    test_file = data_dir / "output_file.txt"
    test_file.write_text("test content")

    config = {
        "configuration": {"data_directory": str(data_dir)},
    }
    integration_test = [
        {"path_descriptor": "data_directory", "file": "output_file.txt", "expected_output": {}}
    ]

    validate_output._validate_output_path_and_file(config, integration_test)
    mock_check_output.assert_called_once()


def test_validate_simtel_cfg_files_with_version_glob_resolution(mocker, tmp_test_directory):
    tmp_dir = Path(str(tmp_test_directory))
    output_path = tmp_dir / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    model_dir = output_path / "model" / "6.0.2"
    model_dir.mkdir(parents=True, exist_ok=True)

    reference_file = Path(
        "tests/resources/sim_telarray_configurations/CTA-North-LSTN-01-6.0_test.cfg"
    )

    test_cfg_content = reference_file.read_text()
    # The expected filename has version 6.0 and label appended, but actual has 6.0.2
    test_file_actual = model_dir / "CTA-North-LSTN-01-6.0.2_test_label.cfg"
    test_file_actual.write_text(test_cfg_content)

    config = {
        "configuration": {
            "output_path": str(output_path),
            "model_version": "6.0.2",
            "label": "test_label",
        },
    }

    validate_output._validate_simtel_cfg_files(config, str(reference_file))


def test_try_resolve_version_path_with_invalid_version(tmp_test_directory):
    """Test _try_resolve_version_path when model_version has less than 2 parts."""
    tmp_dir = Path(str(tmp_test_directory))
    output_path = tmp_dir / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    # Model version with only one part should return None
    result = validate_output._try_resolve_version_path(output_path, "6/file.txt", "6")
    assert result is None


def test_validate_output_path_and_file_without_model_version_in_config(
    tmp_test_directory, mock_check_output
):
    """Test _validate_output_path_and_file when model_version is not in config."""
    tmp_dir = Path(str(tmp_test_directory))
    data_dir = tmp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    test_file = data_dir / "simple_file.txt"
    test_file.write_text("test content")

    # Config without model_version
    config = {
        "configuration": {"data_directory": str(data_dir)},
    }
    integration_test = [
        {"path_descriptor": "data_directory", "file": "simple_file.txt", "expected_output": {}}
    ]

    # This should use the else branch on line 198
    validate_output._validate_output_path_and_file(config, integration_test)
    mock_check_output.assert_called_once()


def test_validate_output_path_and_file_with_model_version_in_config(
    tmp_test_directory, mock_check_output
):
    """Test _validate_output_path_and_file when model_version IS in config."""
    tmp_dir = Path(str(tmp_test_directory))
    data_dir = tmp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create version directory
    version_dir = data_dir / "6.0.2"
    version_dir.mkdir(parents=True, exist_ok=True)
    test_file = version_dir / "output.txt"
    test_file.write_text("test content")

    # Config WITH model_version
    config = {
        "configuration": {
            "data_directory": str(data_dir),
            "model_version": "6.0.2",
        },
    }
    integration_test = [
        {"path_descriptor": "data_directory", "file": "6.0.2/output.txt", "expected_output": {}}
    ]

    # This should use the if branch and call _resolve_output_file_path (line 194-196)
    validate_output._validate_output_path_and_file(config, integration_test)
    mock_check_output.assert_called_once()


def test_normalize_model_version_with_string():
    """Test _normalize_model_version with a string."""
    assert validate_output._normalize_model_version("6.0.2") == "6.0.2"


def test_normalize_model_version_with_list():
    """Test _normalize_model_version with a list returns None (skip version resolution)."""
    assert validate_output._normalize_model_version(["6.0.2", "6.0.1"]) is None


def test_normalize_model_version_with_empty_list():
    """Test _normalize_model_version with an empty list returns None."""
    assert validate_output._normalize_model_version([]) is None


def test_validate_output_path_and_file_with_list_model_version(
    tmp_test_directory, mock_check_output
):
    """Test _validate_output_path_and_file when model_version is a list.

    When model_version is a list like ['6.0', '6.1'], the code should try
    to resolve files with each version until one matches and exists.
    """
    tmp_dir = Path(str(tmp_test_directory))
    data_dir = tmp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create a file with version 6.0.2 (patch version resolved from 6.0)
    test_file = data_dir / "output_6.0.2_data.txt"
    test_file.write_text("test content")

    # Config with model_version as list
    config = {
        "configuration": {
            "data_directory": str(data_dir),
            "model_version": ["6.0", "6.1"],  # List of minor versions
        },
    }
    integration_test = [
        {
            "path_descriptor": "data_directory",
            "file": "output_6.0_data.txt",  # Filename with minor version (will resolve to 6.0.2)
            "expected_output": {},
        }
    ]

    # Should try each version in list and find the file with 6.0.2
    validate_output._validate_output_path_and_file(config, integration_test)
    mock_check_output.assert_called_once()


def test_validate_output_path_and_file_with_list_multiple_files(
    tmp_test_directory, mock_check_output
):
    """Test that all files with different versions from the list are validated.

    Simulates the real integration test scenario where model_version=['6.0', '6.1']
    and there are separate output files for each version.
    """
    tmp_dir = Path(str(tmp_test_directory))
    data_dir = tmp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create files with exact version names (no patch resolution needed)
    file_6_0 = data_dir / "gamma_run001_6.0_output.txt"
    file_6_1 = data_dir / "gamma_run001_6.1_output.txt"
    file_6_0.write_text("6.0 content")
    file_6_1.write_text("6.1 content")

    # Config with model_version as list (like in integration test)
    config = {
        "configuration": {
            "data_directory": str(data_dir),
            "model_version": ["6.0", "6.1"],
        },
    }

    # Test files list exactly as in integration test config
    integration_test = [
        {
            "path_descriptor": "data_directory",
            "file": "gamma_run001_6.0_output.txt",
            "expected_output": {},
        },
        {
            "path_descriptor": "data_directory",
            "file": "gamma_run001_6.1_output.txt",
            "expected_output": {},
        },
    ]

    # Should validate both files
    validate_output._validate_output_path_and_file(config, integration_test)
    assert mock_check_output.call_count == 2


def test_extract_version_from_filename_with_underscore_separator():
    """Test extracting version from filename with underscore separator."""
    assert validate_output._extract_version_from_filename("file_6.0_name.txt") == "6.0"
    assert validate_output._extract_version_from_filename("file_6.1_name.txt") == "6.1"
    assert validate_output._extract_version_from_filename("config_1.2_test.cfg") == "1.2"


def test_extract_version_from_filename_with_slash_separator():
    """Test extracting version from path with slash separator."""
    assert validate_output._extract_version_from_filename("6.0/file.txt") == "6.0"
    assert validate_output._extract_version_from_filename("model/6.1/config.cfg") == "6.1"
    assert validate_output._extract_version_from_filename("path/to/1.5/data.json") == "1.5"


def test_extract_version_from_filename_no_version():
    """Test extracting version from filename without version."""
    assert validate_output._extract_version_from_filename("file_name.txt") is None
    assert validate_output._extract_version_from_filename("config.cfg") is None
    assert validate_output._extract_version_from_filename("path/to/file.json") is None


def test_extract_version_from_filename_patch_version():
    """Test that patch version is ignored (only MAJOR.MINOR extracted)."""
    assert validate_output._extract_version_from_filename("file_6.0.2_name.txt") == "6.0"
    assert validate_output._extract_version_from_filename("6.1.5/file.txt") == "6.1"
    assert validate_output._extract_version_from_filename("config_1.2.3_test.cfg") == "1.2"


def test_extract_version_from_filename_large_version_numbers():
    """Test extracting version with large version numbers."""
    assert validate_output._extract_version_from_filename("file_123.456_name.txt") == "123.456"
    assert validate_output._extract_version_from_filename("9999.8888/file.txt") == "9999.8888"


def test_extract_version_from_filename_complex_path():
    """Test extracting version from complex file paths."""
    filename = "gamma_run000001_za40deg_azm180deg_North_alpha_6.0.2_check_output.log"
    assert validate_output._extract_version_from_filename(filename) == "6.0"

    filename = "CTA-North-LSTN-01_6.0.2_test_label.cfg"
    assert validate_output._extract_version_from_filename(filename) == "6.0"


def test_extract_version_from_filename_first_match():
    """Test that only the first version match is extracted."""
    assert validate_output._extract_version_from_filename("file_6.0_name_7.1_data.txt") == "6.0"
    assert validate_output._extract_version_from_filename("1.2/path/3.4/file.txt") == "1.2"


def test_try_resolve_single_version_path(tmp_test_directory):
    """Test _try_resolve_single_version_path function."""
    tmp_dir = Path(str(tmp_test_directory))
    base_path = tmp_dir / "base"
    base_path.mkdir(parents=True, exist_ok=True)

    # Create version directory with patch
    version_dir = base_path / "6.0.2"
    version_dir.mkdir(parents=True, exist_ok=True)
    test_file = version_dir / "test.md"
    test_file.write_text("test content")

    # Should find file in 6.0.* directory when asked for 6.0
    result = validate_output._try_resolve_single_version_path(base_path, "6.0/test.md", "6.0")
    assert result == test_file

    # Should not find file for non-existent version
    result = validate_output._try_resolve_single_version_path(base_path, "7.0/test.md", "7.0")
    assert result is None


def test_try_resolve_version_path_with_list(tmp_test_directory):
    """Test _try_resolve_version_path with list of versions."""
    tmp_dir = Path(str(tmp_test_directory))
    base_path = tmp_dir / "base"
    base_path.mkdir(parents=True, exist_ok=True)

    # Create version directory
    version_dir = base_path / "6.1.0"
    version_dir.mkdir(parents=True, exist_ok=True)
    test_file = version_dir / "test.md"
    test_file.write_text("test content")

    # Should find file with second version in list
    result = validate_output._try_resolve_version_path(base_path, "6.1/test.md", ["6.0", "6.1"])
    assert result == test_file

    # Should return None if no version matches
    result = validate_output._try_resolve_version_path(base_path, "7.0/test.md", ["7.0", "7.1"])
    assert result is None


def test_try_resolve_version_path_with_none(tmp_test_directory):
    """Test _try_resolve_version_path with None model_version (extract from filename)."""
    tmp_dir = Path(str(tmp_test_directory))
    base_path = tmp_dir / "base"
    base_path.mkdir(parents=True, exist_ok=True)

    # Create version directory
    version_dir = base_path / "6.0.2"
    version_dir.mkdir(parents=True, exist_ok=True)
    test_file = version_dir / "test.md"
    test_file.write_text("test content")

    # Should extract version from filename and find file
    result = validate_output._try_resolve_version_path(base_path, "6.0/test.md", None)
    assert result == test_file

    # Should return None if no version in filename
    result = validate_output._try_resolve_version_path(base_path, "noversion/test.md", None)
    assert result is None


def test_try_resolve_version_path_invalid_version_format(tmp_test_directory):
    """Test _try_resolve_version_path with version having less than 2 parts (lines 205-209)."""
    tmp_dir = Path(str(tmp_test_directory))
    result = validate_output._try_resolve_version_path(tmp_dir, "file.txt", "6")
    assert result is None

    result = validate_output._try_resolve_version_path(tmp_dir, "file.txt", "")
    assert result is None


def test_resolve_file_path_with_versions_fallback_to_direct_path(tmp_test_directory):
    """Test _resolve_file_path_with_versions when no version resolves (lines 291-292)."""
    tmp_dir = Path(str(tmp_test_directory))
    output_path = tmp_dir / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    # No matching version files exist
    result = validate_output._resolve_file_path_with_versions(
        output_path, "file.txt", ["6.0", "6.1"]
    )

    # Should return direct path as fallback
    assert result == output_path / "file.txt"


def test_resolve_file_path_with_versions_with_single_version(tmp_test_directory):
    """Test _resolve_file_path_with_versions with single version string."""
    tmp_dir = Path(str(tmp_test_directory))
    output_path = tmp_dir / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    version_dir = output_path / "6.0.2"
    version_dir.mkdir(parents=True, exist_ok=True)
    test_file = version_dir / "test.txt"
    test_file.write_text("test content")

    # Single version should also work (not just lists)
    result = validate_output._resolve_file_path_with_versions(output_path, "6.0.2/test.txt", "6.0")
    assert result == test_file


def test_resolve_file_path_with_versions_oserror_handling(tmp_test_directory, mocker):
    """Test _resolve_file_path_with_versions handles OSError gracefully."""
    tmp_dir = Path(str(tmp_test_directory))
    output_path = tmp_dir / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    # Mock _resolve_output_file_path to raise OSError
    mocker.patch(
        "simtools.testing.validate_output._resolve_output_file_path",
        side_effect=OSError("Permission denied"),
    )

    # Should catch OSError and return direct path
    result = validate_output._resolve_file_path_with_versions(output_path, "file.txt", ["6.0"])
    assert result == output_path / "file.txt"


def test_resolve_file_path_with_versions_valueerror_handling(tmp_test_directory, mocker):
    """Test _resolve_file_path_with_versions handles ValueError gracefully."""
    tmp_dir = Path(str(tmp_test_directory))
    output_path = tmp_dir / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    # Mock _resolve_output_file_path to raise ValueError
    mocker.patch(
        "simtools.testing.validate_output._resolve_output_file_path",
        side_effect=ValueError("Invalid version"),
    )

    # Should catch ValueError and return direct path
    result = validate_output._resolve_file_path_with_versions(output_path, "file.txt", ["6.0"])
    assert result == output_path / "file.txt"


def test_validate_output_files(mocker, output_path):
    """Test _validate_output_files function."""
    mock_validate_reference = mocker.patch(
        "simtools.testing.validate_output._validate_reference_output_file"
    )
    mock_validate_path = mocker.patch(
        "simtools.testing.validate_output._validate_output_path_and_file"
    )
    mock_validate_model_param = mocker.patch(
        "simtools.testing.validate_output._validate_model_parameter_json_file"
    )

    config = {"configuration": {"output_path": output_path}}
    integration_test = {
        "reference_output_file": "/path/to/ref.txt",
        "test_output_files": [{"file": "test.txt"}],
        "output_file": "output.txt",
        "model_parameter_validation": {
            "reference_parameter_name": "param",
            "parameter_file": "param.json",
            "tolerance": 1.0e-5,
        },
    }

    validate_output._validate_output_files(config, integration_test, db_config=None)

    mock_validate_reference.assert_called_once_with(config, integration_test)
    assert mock_validate_path.call_count == 2
    mock_validate_model_param.assert_called_once()


def test_validate_model_parameter_json_file_with_version_in_path(mocker, output_path):
    """Test _validate_model_parameter_json_file extracts version from path."""
    mock_db_handler = mocker.patch("simtools.db.db_handler.DatabaseHandler")
    mock_collect_data = mocker.patch("simtools.io.ascii_handler.collect_data_from_file")
    mocker.patch(
        "simtools.testing.validate_output._compare_value_from_parameter_dict", return_value=True
    )

    mock_db_instance = mock_db_handler.return_value
    mock_db_instance.get_model_parameter.return_value = {"param": {"value": [1.0]}}
    mock_collect_data.return_value = {"value": [1.0]}

    config = {
        "configuration": {
            "output_path": output_path,
            "telescope": "telescope",
            "model_version": ["6.0", "6.1"],  # List, but should extract from path
            "site": "site",
        }
    }
    model_parameter_validation = {
        "reference_parameter_name": "param",
        "parameter_file": "param_6.0.2_data.json",  # Version in filename
        "tolerance": 1.0e-5,
    }

    validate_output._validate_model_parameter_json_file(
        config, model_parameter_validation, db_config=None
    )

    # Should use version extracted from filename (6.0)
    mock_db_instance.get_model_parameter.assert_called_once_with(
        parameter="param",
        site="site",
        array_element_name="telescope",
        model_version="6.0",  # Extracted from filename, not from config list
    )

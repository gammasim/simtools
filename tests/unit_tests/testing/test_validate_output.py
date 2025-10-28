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
    assert resolved == test_file1


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

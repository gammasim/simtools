import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from astropy.table import Table

from simtools.testing import validate_output
from simtools.testing.validate_output import (
    _validate_output_path_and_file,
    _versions_match,
    validate_application_output,
)

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


@pytest.mark.parametrize(
    ("content", "content_diff", "should_match"),
    [
        ({"key": 1, "value": "1.23 4.56 7.89", "schema_version": "1.0.0"}, None, True),
        (
            {"key": 1, "value": "1.23 4.56 7.89", "schema_version": "1.0.0"},
            {"key": 2, "value": "1.23 4.56 7.80", "schema_version": "2.0.0"},
            False,
        ),
    ],
)
def test_compare_json_files_float_strings(
    create_json_file, file_name, content, content_diff, should_match
):
    """Test JSON comparison with float strings."""
    file1 = create_json_file(file_name(1, "json"), content)
    file2 = create_json_file(
        file_name(2, "json"), content if content_diff is None else content_diff
    )

    assert validate_output.compare_json_or_yaml_files(file1, file2) == should_match


@pytest.mark.parametrize(
    ("content1", "content2", "should_match"),
    [
        ({"key": 1, "value": 5}, {"key": 1, "value": 5, "extra": "extra"}, False),
        ({"key": 1, "value": 5}, {"different_key": 1, "value": 5}, False),
        ({"key": 1, "value": 5}, {"key": 2, "value": 7}, False),
    ],
)
def test_compare_json_files_unequal_dicts(
    create_json_file, file_name, content1, content2, should_match
):
    """Test JSON comparison with unequal dictionaries."""
    file1 = create_json_file(file_name(1, "json"), content1)
    file2 = create_json_file(file_name(2, "json"), content2)
    assert validate_output.compare_json_or_yaml_files(file1, file2) == should_match


@pytest.mark.parametrize(
    ("file_type", "create_func"),
    [("json", "create_json_file"), ("yaml", "create_yaml_file")],
)
def test_compare_files_equal_integers(file_type, create_func, file_name, request):
    """Test JSON/YAML comparison with equal integers."""
    create_file = request.getfixturevalue(create_func)
    content = {"key": 1, "value": 5}
    file1 = create_file(file_name(1, file_type), content)
    file2 = create_file(file_name(2, file_type), content)

    assert validate_output.compare_json_or_yaml_files(file1, file2)

    content3 = {"key": 2, "value": 7}
    file3 = create_file(file_name(3, file_type), content3)
    assert not validate_output.compare_json_or_yaml_files(file1, file3)


@pytest.mark.parametrize(
    ("data1", "data2", "tolerance", "should_match"),
    [
        ({"key": 1, "value": 5.5}, {"key": 1, "value": 5.5}, 1e-5, True),
        ({"key": 1, "value": 5.5}, {"key": 1, "value": 5.75}, 1e-5, False),
        ({"key": 1, "value": 5.5}, {"key": 1, "value": 5.75}, 0.5, True),
        ({"key": 1, "value": [5.5, 10.5]}, {"key": 1, "value": [5.5, 10.5]}, 1e-5, True),
        ({"key": 1, "value": [5.5, 10.5]}, {"key": 1, "value": 5.75}, 1e-5, False),
        ({"key": 1, "value": [5.5, 10.5]}, {"key": 1, "value": [5.75, 10.75]}, 0.5, True),
    ],
)
def test_compare_json_files_floats(
    create_json_file, file_name, data1, data2, tolerance, should_match
):
    """Test JSON comparison with floats and lists of floats."""
    file1 = create_json_file(file_name(1, "json"), data1)
    file2 = create_json_file(file_name(2, "json"), data2)

    assert (
        validate_output.compare_json_or_yaml_files(file1, file2, tolerance=tolerance)
        == should_match
    )


def test_compare_yaml_files_float_strings(create_yaml_file, file_name):
    """Test YAML comparison with float strings."""
    content = {"key": 1, "value": "1.23 4.56 7.89"}
    file1 = create_yaml_file(file_name(1, "yaml"), content)
    file2 = create_yaml_file(file_name(2, "yaml"), content)

    assert validate_output.compare_json_or_yaml_files(file1, file2)

    content3 = {"key": 1, "value": "1.23 4.56 7.80"}
    file3 = create_yaml_file(file_name(3, "yaml"), content3)
    assert not validate_output.compare_json_or_yaml_files(file1, file3)

    assert validate_output.compare_json_or_yaml_files(file1, file3, tolerance=0.5)


def test_compare_json_files_nested_dicts_with_values(create_json_file, file_name):
    content = {
        "meta": {"tel": "LSTN-01", "zen": 20.0},
        "efficiency": {"value": 0.2748, "description": "Camera nominal efficiency with gaps"},
    }
    file1 = create_json_file(file_name(1, "json"), content)
    file2 = create_json_file(file_name(2, "json"), content)

    assert validate_output.compare_json_or_yaml_files(file1, file2)

    content3 = {
        "meta": {"tel": "LSTN-01", "zen": 20.0},
        "efficiency": {"value": 0.2850, "description": "Camera nominal efficiency with gaps"},
    }
    file3 = create_json_file(file_name(3, "json"), content3)
    assert not validate_output.compare_json_or_yaml_files(file1, file3)

    assert validate_output.compare_json_or_yaml_files(file1, file3, tolerance=0.05)


@pytest.mark.parametrize(
    ("content1", "content2", "tolerance", "should_match"),
    [
        ({"col1": [1.1, 2.2, 3.3], "col2": [4.4, 5.5, 6.6]}, None, 1e-5, True),
        (
            {"col1": [1.1, 2.2, 3.3], "col2": [4.4, 5.5, 6.6]},
            {"col1": [1.1, 2.2], "col2": [4.4, 5.5]},
            1e-5,
            False,
        ),
        (
            {"col1": [1.1001, 2.2001, 3.3001], "col2": [4.4001, 5.5001, 6.6001]},
            {"col1": [1.1, 2.2, 3.3], "col2": [4.4, 5.5, 6.6]},
            1e-3,
            True,
        ),
        (
            {"col1": [1.1, 2.2, 3.3], "col2": [4.4, 5.5, 6.6]},
            {"col1": [10.1, 20.2, 30.3], "col2": [40.4, 50.5, 60.6]},
            1e-5,
            False,
        ),
    ],
)
def test_compare_ecsv_files(
    create_ecsv_file, file_name, content1, content2, tolerance, should_match
):
    """Test ECSV file comparison with various scenarios."""
    file1 = create_ecsv_file(file_name(1, "ecsv"), content1)
    file2 = create_ecsv_file(file_name(2, "ecsv"), content2 if content2 else content1)

    assert validate_output.compare_ecsv_files(file1, file2, tolerance=tolerance) == should_match


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
        "configuration": {"output_path": output_path},
        "integration_tests": [{"expected_output": "expected_output"}],
    }
    integration_test = [
        {
            "path_descriptor": "output_path",
            "file": "output_file.simtel.zst",
            "expected_output": {},
        }
    ]

    validate_output._validate_output_path_and_file(config, integration_test)

    mock_path_exists.assert_called()
    mock_check_output.assert_called_once_with(
        Path(config["configuration"]["output_path"]).joinpath(integration_test[0]["file"]),
        {
            "path_descriptor": "output_path",
            "file": "output_file.simtel.zst",
            "expected_output": {},
        },
    )

    wrong_integration_test = [
        {"path_descriptor": "wrong_path", "file": "output_file.simtel.zst", "expected_output": {}}
    ]
    with pytest.raises(
        KeyError, match=r"Path wrong_path not found in integration test configuration."
    ):
        validate_output._validate_output_path_and_file(config, wrong_integration_test)


def test_validate_application_output_no_integration_tests(mocker, output_path):
    """Test validate_application_output when no integration tests are present."""
    config = {"configuration": {"output_path": output_path}}
    mock_logger_info = mocker.patch("simtools.testing.validate_output._logger.info")

    result = validate_output.validate_application_output(config)

    mock_logger_info.assert_not_called()
    assert result is None


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
    dummy_dir_content = [Path("dummy_file")]
    with (
        patch("pathlib.Path.exists", return_value=False),
        patch("pathlib.Path.iterdir", return_value=dummy_dir_content),
    ):
        with pytest.raises(
            AssertionError,
            match=r"Output file /path/to/output/not_there does not exist. Directory contents: \[PosixPath\('dummy_file'\)\]",
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
    file1 = Path("tests/resources/sim_telarray_configurations/6.0.2/CTA-North-LSTN-01_test.cfg")
    file2 = Path("tests/resources/sim_telarray_configurations/6.0.2/CTA-North-LSTN-01_test.cfg")

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
    """Test validation of simtel cfg files."""
    mock_compare = mocker.patch(
        "simtools.testing.validate_output._compare_simtel_cfg_files", return_value=True
    )
    config = {
        "configuration": {
            "output_path": PATH_TO_OUTPUT,
            "model_version": "3.4.5",
            "label": "label",
        },
        "integration_tests": [{"test_simtel_cfg_files": test_path}],
    }
    validate_output._validate_simtel_cfg_files(config, test_path)

    mock_compare.assert_called()


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

    validate_output._validate_model_parameter_json_file(config, model_parameter_validation)

    mock_db_handler.assert_called_once_with()
    mock_db_instance.get_model_parameter.assert_called_once_with(
        parameter="reference_param",
        site="test_site",
        array_element_name="test_telescope",
        model_version="1.0.0",
    )
    mock_collect_data_from_file.assert_called_once_with(
        Path(output_path) / "test_telescope" / TEST_PARAM_JSON
    )
    mock_compare_value.assert_called_once_with([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1.0e-5, 1.0)


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
        validate_output._validate_model_parameter_json_file(config, model_parameter_validation)

    mock_db_handler.assert_called_once_with()
    mock_db_instance.get_model_parameter.assert_called_once_with(
        parameter="reference_param",
        site="test_site",
        array_element_name="test_telescope",
        model_version="1.0.0",
    )
    mock_collect_data_from_file.assert_called_once_with(
        Path(output_path) / "test_telescope" / TEST_PARAM_JSON
    )
    mock_compare_value.assert_called_once_with([1.1, 2.1, 3.1], [1.0, 2.0, 3.0], 1.0e-5, 1.0)


def test_versions_match_semantics():
    # No CLI filter, always validate
    assert _versions_match(None, "6.0.0") is True

    # Exact match
    assert _versions_match("6.0.0", "6.0.0") is True

    # No overlap
    assert _versions_match("6.0.0", "7.0.0") is False

    # Overlap where CLI provides multiple
    assert _versions_match(["6.0.0", "7.0.0"], "7.0.0") is True

    # Overlap where config provides multiple
    assert _versions_match(["6.0.0"], ["7.0.0", "6.0.0"]) is True

    # No overlap where config provides multiple
    assert _versions_match("8.0.0", ["7.0.0", "6.0.0"]) is False


def test_validate_output_path_and_file_routes_by_suffix(tmp_path: Path):
    # Create three test files with expected suffixes
    simtel = tmp_path / "out.simtel.zst"
    tarlog = tmp_path / "logs.log_hist.tar.gz"
    plain = tmp_path / "logfile.log"
    for f in (simtel, tarlog, plain):
        f.write_text("content", encoding="utf-8")

    cfg = {"configuration": {"output_path": str(tmp_path)}}
    file_tests = [
        {"path_descriptor": "output_path", "file": simtel.name},
        {"path_descriptor": "output_path", "file": tarlog.name, "expected_log_output": {}},
        {"path_descriptor": "output_path", "file": plain.name, "expected_log_output": {}},
    ]

    with (
        patch(
            "simtools.testing.assertions.check_output_from_sim_telarray", return_value=True
        ) as m_simtel,
    ):
        _validate_output_path_and_file(cfg, file_tests)

        m_simtel.assert_called_once()


def test_validate_output_path_and_file_missing_raises(tmp_path: Path):
    cfg = {"configuration": {"output_path": str(tmp_path)}}
    missing = "does_not_exist.log"
    with pytest.raises(AssertionError, match=r"Output file .* does not exist"):
        _validate_output_path_and_file(cfg, [{"path_descriptor": "output_path", "file": missing}])


def test_validate_application_output_gating_calls(tmp_path: Path):
    cfg = {
        "configuration": {"output_path": str(tmp_path), "output_file": "x"},
        "integration_tests": [{"output_file": "x"}],
    }

    with (
        patch("simtools.testing.validate_output._validate_output_files") as m_validate,
        patch("simtools.testing.validate_output._test_simtel_cfg_files"),
    ):
        # No CLI filter, should call validation
        validate_application_output(cfg, from_command_line=None, from_config_file="6.0.0")
        assert m_validate.called

    with (
        patch("simtools.testing.validate_output._validate_output_files") as m_validate,
        patch("simtools.testing.validate_output._test_simtel_cfg_files"),
    ):
        # CLI filter not matching, should skip validations
        validate_application_output(cfg, from_command_line="7.0.0", from_config_file="6.0.0")
        m_validate.assert_not_called()


@pytest.mark.parametrize(
    ("data1", "data2", "tolerance", "is_value_field", "should_match"),
    [
        ({"key": 1, "value": 5.5}, {"key": 1, "value": 5.5}, 1e-5, False, True),
        ({"key": 1, "value": 5.5}, {"different_key": 1, "value": 5.5}, 1e-5, False, False),
        ({"value": 5.5}, {"value": 5.50001}, 1e-3, False, True),
        ({"value": 5.5}, {"value": 5.75}, 1e-5, False, False),
        ({"values": [1.1, 2.2, 3.3]}, {"values": [1.1, 2.2, 3.3]}, 1e-5, False, True),
        ({"values": [1.1, 2.2, 3.3]}, {"values": [1.1, 2.2]}, 1e-5, False, False),
        (
            {"meta": {"tel": "LSTN-01"}, "efficiency": {"value": 0.2748}},
            {"meta": {"tel": "LSTN-01"}, "efficiency": {"value": 0.2748}},
            1e-5,
            False,
            True,
        ),
        (
            {"meta": {"tel": "LSTN-01"}, "efficiency": {"value": 0.2748}},
            {"meta": {"tel": "LSTN-01"}, "efficiency": {"value": 0.2850}},
            1e-5,
            False,
            False,
        ),
        (
            {"meta": {"tel": "LSTN-01"}, "efficiency": {"value": 0.2748}},
            {"meta": {"tel": "LSTN-01"}, "efficiency": {"value": 0.2850}},
            0.05,
            False,
            True,
        ),
        (
            {"description": "test string", "value": 5.5},
            {"description": "test string", "value": 5.50001},
            1e-3,
            False,
            True,
        ),
        ({"description": "test string"}, {"description": "different string"}, 1e-5, False, False),
        (5.5, 5.50001, 1e-3, True, True),
        (5.5, 5.75, 1e-5, True, False),
        ((1.1, 2.2, 3.3), [1.1, 2.2, 3.3], 1e-5, False, True),
        ({"items": [1, "string", 3.3]}, {"items": [1, "string", 3.3]}, 1e-5, False, True),
        ({"value": "1.23 4.56 7.89"}, {"value": "1.23 4.56 7.89"}, 1e-5, False, True),
        ({"value": "1.23 4.56 7.89"}, {"value": "1.23 4.56 7.90"}, 1e-5, False, False),
    ],
)
def test_compare_nested_dicts_with_tolerance(data1, data2, tolerance, is_value_field, should_match):
    """Test nested dict comparison with various scenarios."""
    assert (
        validate_output._compare_nested_dicts_with_tolerance(
            data1, data2, tolerance=tolerance, is_value_field=is_value_field
        )
        == should_match
    )

import json
import logging
from collections import UserDict
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from astropy import units as u
from astropy.table import Table

from simtools.constants import TEST_RESOURCES_GENERATED
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


@pytest.mark.parametrize("resource_dir", ["generated", "downloaded"])
def test_compare_json_files_resource_paths(
    create_json_file, file_name, tmp_test_directory, resource_dir
):
    """Test comparison of equivalent resource paths with different roots."""
    resource_file = Path(resource_dir) / "model_parameters" / "file.lis"
    configured_resource_root = tmp_test_directory / "configured-resources"

    content = {"meta": {"nsb": str(resource_file)}}
    content_absolute = {"meta": {"nsb": str(configured_resource_root / resource_file)}}
    content_other = {
        "meta": {
            "nsb": str(configured_resource_root / resource_dir / "model_parameters" / "other.lis")
        }
    }

    file1 = create_json_file(file_name(1, "json"), content)
    file2 = create_json_file(file_name(2, "json"), content_absolute)
    file3 = create_json_file(file_name(3, "json"), content_other)

    assert validate_output.compare_json_or_yaml_files(file1, file2)
    assert not validate_output.compare_json_or_yaml_files(file1, file3)


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
        Path(integration_test["reference_output_file"]),
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
    file1 = Path(f"{TEST_RESOURCES_GENERATED}/sim_telarray_configurations/7.0.0/CTAO-LSTN-01.cfg")
    file2 = Path(f"{TEST_RESOURCES_GENERATED}/sim_telarray_configurations/7.0.0/CTAO-LSTN-01.cfg")

    assert validate_output._compare_simtel_cfg_files(file1, file2)

    with open(file1) as f1:
        lines1 = f1.readlines()

    # additional metadata line should be ignored here
    file3 = tmp_test_directory / "file3.cfg"
    with open(file3, "a") as f3:
        f3.write("".join(lines1))
        f3.write("metaparam telescope set simtools_version = changed\n")
    assert validate_output._compare_simtel_cfg_files(file1, file3)

    # change of values
    file4 = tmp_test_directory / "file4.cfg"
    with open(file4, "a") as f3:
        f3.write("".join(lines1).replace("1", "2"))
    assert not validate_output._compare_simtel_cfg_files(file1, file4)

    # additional control line should still fail
    file5 = tmp_test_directory / "file5.cfg"
    with open(file5, "a") as f5:
        f5.write("".join(lines1))
        f5.write("# include <Extra.cfg>\n")
    assert not validate_output._compare_simtel_cfg_files(file1, file5)


def test_validate_simtel_cfg_files(mocker, test_path):
    """Test validation of simtel cfg files."""
    mock_run_number = mocker.patch(
        "simtools.testing.validate_output.file_info.get_corsika_run_number", return_value=7
    )
    mock_compare = mocker.patch(
        "simtools.testing.validate_output._compare_simtel_cfg_files", return_value=True
    )
    config = {
        "configuration": {
            "output_path": PATH_TO_OUTPUT,
            "model_version": "3.4.5",
            "label": "label",
            "run_number": 42,
            "run_number_offset": 1,
            "corsika_file": test_path,
        },
        "integration_tests": [{"test_simtel_cfg_files": test_path}],
    }
    validate_output._validate_simtel_cfg_files(config, test_path)

    mock_run_number.assert_called_once_with(test_path)
    assert mock_compare.call_args.args[1].parent == Path(PATH_TO_OUTPUT) / "model/run000007/3.4.5"


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


def test_validate_output_path_and_file_routes_by_suffix(tmp_test_directory):
    simtel = Path(str(tmp_test_directory)) / "out.simtel.zst"
    tarlog = Path(str(tmp_test_directory)) / "logs.log_hist.tar.gz"
    plain = Path(str(tmp_test_directory)) / "logfile.log"
    for f in (simtel, tarlog, plain):
        f.write_text("content", encoding="utf-8")

    cfg = {"configuration": {"output_path": str(tmp_test_directory)}}
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


def test_validate_output_path_and_file_missing_raises(tmp_test_directory):
    cfg = {"configuration": {"output_path": str(tmp_test_directory)}}
    missing = "does_not_exist.log"
    with pytest.raises(AssertionError, match=r"Output file .* does not exist"):
        _validate_output_path_and_file(cfg, [{"path_descriptor": "output_path", "file": missing}])


def test_validate_application_output_gating_calls(tmp_test_directory):
    cfg = {
        "configuration": {"output_path": str(tmp_test_directory), "output_file": "x"},
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


def _semantic_table(path, metadata=None):
    """Write a small ECSV table used by declarative output-validation tests."""
    table = Table(
        {
            "run_number": [1, 2],
            "value": [1.0, 2.0],
            "label": ["a", "b"],
        }
    )
    table["value"].unit = u.m
    table.meta = metadata if metadata is not None else {"summary": {"rows": 2, "total": 3.0}}
    table.write(path, format="ascii.ecsv", overwrite=True)


def _semantic_schema(path):
    """Write the data-product schema for the semantic table."""
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "0.1.0",
                "data": [
                    {
                        "type": "data_table",
                        "table_columns": [
                            {"name": "run_number", "required": True, "type": "int64"},
                            {
                                "name": "value",
                                "required": True,
                                "type": "float64",
                                "unit": "m",
                            },
                            {"name": "label", "required": True, "type": "string"},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _semantic_config(output_path, rule):
    """Build a configuration containing one semantic output rule."""
    return {
        "configuration": {"output_path": str(output_path)},
        "integration_tests": [{"output_validation": [rule]}],
    }


def _semantic_rule(output_file, schema_file):
    """Build the representative ECSV output-validation rule."""
    return {
        "name": "semantic_table",
        "path_descriptor": "output_path",
        "file": output_file.name,
        "data_product_schema": str(schema_file),
        "minimum_rows": 1,
        "unique_columns": ["run_number"],
        "columns": {
            "value": {"range": {"minimum": 0.0, "maximum": 3.0, "unit": "m"}},
            "label": {"allowed_values": ["a", "b"]},
        },
        "metadata": {
            "required_keys": ["summary"],
            "row_count": "summary.rows",
            "column_sums": {"value": "summary.total"},
        },
    }


def test_declarative_table_validation_passes(tmp_test_directory):
    """Validate schema, rows, domains, uniqueness, and metadata summaries."""
    tmp_test_directory = Path(tmp_test_directory)
    output_file = tmp_test_directory / "table.ecsv"
    schema_file = tmp_test_directory / "table.schema.yml"
    _semantic_table(output_file)
    _semantic_schema(schema_file)

    validate_output.validate_application_output(
        _semantic_config(tmp_test_directory, _semantic_rule(output_file, schema_file))
    )


def test_has_path_supports_mapping_metadata():
    """Accept mapping implementations used for ordered metadata."""
    metadata = UserDict({"summary": UserDict({"rows": 2})})

    assert validate_output._has_path(metadata, "summary.rows")


def test_declarative_table_rejects_empty_and_duplicate_rows(tmp_test_directory):
    """Reject an empty table and duplicate values in a unique column."""
    tmp_test_directory = Path(tmp_test_directory)
    output_file = tmp_test_directory / "table.ecsv"
    schema_file = tmp_test_directory / "table.schema.yml"
    _semantic_schema(schema_file)
    rule = _semantic_rule(output_file, schema_file)
    Table(
        names=["run_number", "value", "label"],
        dtype=["int64", "float64", "str"],
        meta={"summary": {"rows": 0, "total": 0.0}},
    ).write(output_file, format="ascii.ecsv")
    with pytest.raises(AssertionError, match="at least 1 rows"):
        validate_output.validate_application_output(_semantic_config(tmp_test_directory, rule))

    _semantic_table(output_file)
    table = Table.read(output_file, format="ascii.ecsv")
    table["run_number"][1] = table["run_number"][0]
    table.write(output_file, format="ascii.ecsv", overwrite=True)
    with pytest.raises(AssertionError, match="unique values in run_number"):
        validate_output.validate_application_output(_semantic_config(tmp_test_directory, rule))


@pytest.mark.parametrize(
    ("column", "column_rule", "expected"),
    [
        ("label", {"allowed_values": ["x"]}, "allowed values"),
        ("label", {"range": {"minimum": 0.0}}, "numerical range"),
        ("value", {"range": {"minimum": 2.0, "unit": "m"}}, "minimum"),
        ("value", {"range": {"maximum": 1.0, "unit": "m"}}, "maximum"),
    ],
)
def test_declarative_column_validation_failures(tmp_test_directory, column, column_rule, expected):
    """Reject workflow-specific column domains and ranges."""
    tmp_test_directory = Path(tmp_test_directory)
    output_file = tmp_test_directory / "table.ecsv"
    schema_file = tmp_test_directory / "table.schema.yml"
    _semantic_table(output_file)
    _semantic_schema(schema_file)
    rule = _semantic_rule(output_file, schema_file)
    rule["columns"] = {column: column_rule}

    with pytest.raises(AssertionError, match=expected):
        validate_output.validate_application_output(_semantic_config(tmp_test_directory, rule))


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({}, "required metadata key summary"),
        ({"summary": {"rows": 3, "total": 3.0}}, "equals row count"),
        ({"summary": {"rows": 2, "total": 4.0}}, "equals sum of value"),
    ],
)
def test_declarative_metadata_validation_failures(tmp_test_directory, metadata, expected):
    """Reject missing or inconsistent table metadata."""
    tmp_test_directory = Path(tmp_test_directory)
    output_file = tmp_test_directory / "table.ecsv"
    schema_file = tmp_test_directory / "table.schema.yml"
    _semantic_table(output_file, metadata=metadata)
    _semantic_schema(schema_file)

    with pytest.raises(AssertionError, match=expected):
        validate_output.validate_application_output(
            _semantic_config(
                tmp_test_directory,
                _semantic_rule(output_file, schema_file),
            )
        )


def test_declarative_data_product_schema_validation(tmp_test_directory):
    """Reject a table that does not satisfy its data-product schema."""
    tmp_test_directory = Path(tmp_test_directory)
    output_file = tmp_test_directory / "table.ecsv"
    schema_file = tmp_test_directory / "table.schema.yml"
    _semantic_schema(schema_file)
    Table({"other": [1, 2]}).write(output_file, format="ascii.ecsv")

    with pytest.raises(AssertionError, match="data-product schema"):
        validate_output.validate_application_output(
            _semantic_config(
                tmp_test_directory,
                _semantic_rule(output_file, schema_file),
            )
        )


def test_declarative_output_must_exist_and_be_ecsv(tmp_test_directory):
    """Report missing and unparsable output tables consistently."""
    tmp_test_directory = Path(tmp_test_directory)
    output_file = tmp_test_directory / "table.ecsv"
    schema_file = tmp_test_directory / "table.schema.yml"
    _semantic_schema(schema_file)
    rule = _semantic_rule(output_file, schema_file)

    with pytest.raises(AssertionError, match="existing output"):
        validate_output.validate_application_output(_semantic_config(tmp_test_directory, rule))

    output_file.write_text("not an ECSV table\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="parseable ECSV table"):
        validate_output.validate_application_output(_semantic_config(tmp_test_directory, rule))

    _semantic_table(output_file)
    rule["unique_columns"] = ["missing"]
    with pytest.raises(AssertionError, match="valid configured table content"):
        validate_output.validate_application_output(_semantic_config(tmp_test_directory, rule))

    rule["path_descriptor"] = "missing"
    with pytest.raises(KeyError, match="Path missing"):
        validate_output.validate_application_output(_semantic_config(tmp_test_directory, rule))

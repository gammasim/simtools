"""Compare application output to reference output."""

import logging
from pathlib import Path

import numpy as np
from astropy.table import Table

import simtools.utils.general as gen
from simtools.db import db_handler
from simtools.io import ascii_handler
from simtools.testing import assertions

_logger = logging.getLogger(__name__)


def _versions_match(from_command_line, from_config_file):
    """Return True if validations should run for the given versions.

    Behavior:
    - If no version is provided from the command line, run validations.
    - If a filter is provided from the command line, run only when it matches
        the version(s) from the config file.
    """
    if from_command_line is None:
        return True

    # Normalize to collections for comparison
    cmd_versions = from_command_line if isinstance(from_command_line, list) else [from_command_line]
    cfg_versions = from_config_file if isinstance(from_config_file, list) else [from_config_file]

    # Consider a match if any overlap exists
    return any(cv in cmd_versions for cv in cfg_versions)


# Keys to ignore when comparing sim_telarray configuration files
# (e.g., version numbers, system dependent parameters, CORSIKA options)
cfg_ignore_keys = [
    "config_release",
    "Label",
    "simtools_",  # ignore all simtools_ keys - version/build info dependence
]


def validate_application_output(config, from_command_line=None, from_config_file=None):
    """
    Validate application output against expected output.

    Expected output is defined in configuration file.
    Some tests run only if the model version from the command line
    equals the model version from the configuration file.

    Parameters
    ----------
    config: dict
        dictionary with the configuration for the application test.
    from_command_line: str
        Model version from the command line.
    from_config_file: str
        Model version from the configuration file.

    """
    if "integration_tests" not in config:
        return

    for integration_test in config["integration_tests"]:
        _logger.info(f"Testing application output: {integration_test}")
        _logger.debug(
            f"Model version from command line: {from_command_line}, "
            f"from config file: {from_config_file}"
        )

        if _versions_match(from_command_line, from_config_file):
            _validate_output_files(config, integration_test)

            if "file_type" in integration_test:
                assert assertions.assert_file_type(
                    integration_test["file_type"],
                    Path(config["configuration"]["output_path"]).joinpath(
                        config["configuration"]["output_file"]
                    ),
                )
        _test_simtel_cfg_files(config, integration_test, from_command_line, from_config_file)


def _validate_output_files(config, integration_test):
    """Validate output files."""
    if "reference_output_file" in integration_test:
        _validate_reference_output_file(config, integration_test)
    if "test_output_files" in integration_test:
        _validate_output_path_and_file(config, integration_test["test_output_files"])
    if "output_file" in integration_test:
        _validate_output_path_and_file(
            config,
            [{"path_descriptor": "output_path", "file": integration_test["output_file"]}],
        )
    if "model_parameter_validation" in integration_test:
        _validate_model_parameter_json_file(config, integration_test["model_parameter_validation"])


def _test_simtel_cfg_files(config, integration_test, from_command_line, from_config_file):
    """Test simtel cfg files."""
    cfg_files = integration_test.get("test_simtel_cfg_files", {})
    if isinstance(from_command_line, list):
        sources = from_command_line
    elif isinstance(from_config_file, list):
        sources = from_config_file
    else:
        sources = [from_command_line or from_config_file]
    for version in sources:
        cfg = cfg_files.get(version)
        if cfg:
            _validate_simtel_cfg_files(config, cfg)
            break


def _validate_reference_output_file(config, integration_test):
    """Compare with reference output file."""
    assert compare_files(
        integration_test["reference_output_file"],
        Path(config["configuration"]["output_path"]).joinpath(
            config["configuration"]["output_file"]
        ),
        integration_test.get("tolerance", 1.0e-5),
        integration_test.get("test_columns", None),
    )


def _validate_output_path_and_file(config, integration_file_tests):
    """Check if output paths and files exist."""
    for file_test in integration_file_tests:
        try:
            output_path = config["configuration"][file_test["path_descriptor"]]
        except KeyError as exc:
            raise KeyError(
                f"Path {file_test['path_descriptor']} not found in integration test configuration."
            ) from exc

        output_file_path = Path(output_path) / file_test["file"]
        _logger.info(f"Checking path: {output_file_path}")
        try:
            assert output_file_path.exists()
        except AssertionError as exc:
            raise AssertionError(f"Output file {output_file_path} does not exist. ") from exc

        if output_file_path.name.endswith(".simtel.zst"):
            assert assertions.check_output_from_sim_telarray(output_file_path, file_test)
        elif output_file_path.name.endswith(".log_hist.tar.gz"):
            assert assertions.check_simulation_logs(output_file_path, file_test)
        elif output_file_path.suffix == ".log":
            assert assertions.check_plain_log(output_file_path, file_test)


def _validate_model_parameter_json_file(config, model_parameter_validation):
    """
    Validate model parameter json file and compare it with a reference parameter from the database.

    Requires database connection to pull the model parameter for a given telescope or site model.

    Parameters
    ----------
    config: dict
        Configuration dictionary.
    model_parameter_validation: dict
        Dictionary with model parameter validation configuration.

    """
    _logger.info(f"Checking model parameter json file: {model_parameter_validation}")
    db = db_handler.DatabaseHandler()

    reference_parameter_name = model_parameter_validation.get("reference_parameter_name")

    reference_model_parameter = db.get_model_parameter(
        parameter=reference_parameter_name,
        site=config["configuration"].get("site"),
        array_element_name=config["configuration"].get("telescope"),
        model_version=config["configuration"].get("model_version"),
    )
    parameter_file = (
        Path(config["configuration"]["output_path"])
        / config["configuration"].get("telescope")
        / model_parameter_validation["parameter_file"]
    )
    model_parameter = ascii_handler.collect_data_from_file(parameter_file)
    assert _compare_value_from_parameter_dict(
        model_parameter["value"],
        reference_model_parameter[reference_parameter_name]["value"],
        model_parameter_validation["tolerance"],
    )


def compare_files(file1, file2, tolerance=1.0e-5, test_columns=None):
    """
    Compare two files of file type ecsv, json or yaml.

    Parameters
    ----------
    file1: str
        First file to compare
    file2: str
        Second file to compare
    tolerance: float
        Tolerance for comparing numerical values.
    test_columns: list
        List of columns to compare. If None, all columns are compared.

    Returns
    -------
    bool
        True if the files are equal.

    """
    _file1_suffix = Path(file1).suffix
    _file2_suffix = Path(file2).suffix
    _logger.info("Comparing files: %s and %s", file1, file2)
    if _file1_suffix != _file2_suffix:
        raise ValueError(f"File suffixes do not match: {file1} and {file2}")
    if _file1_suffix == ".ecsv":
        return compare_ecsv_files(file1, file2, tolerance, test_columns)
    if _file1_suffix in (".json", ".yaml", ".yml"):
        return compare_json_or_yaml_files(file1, file2, tolerance)

    _logger.warning(f"Unknown file type for files: {file1} and {file2}")
    return False


def compare_json_or_yaml_files(file1, file2, tolerance=1.0e-2):
    """
    Compare two json or yaml files.

    Take into account float comparison for sim_telarray string-embedded floats.
    Allow differences in 'schema_version' field.

    Parameters
    ----------
    file1: str
        First file to compare
    file2: str
        Second file to compare
    tolerance: float
        Tolerance for comparing numerical values.

    Returns
    -------
    bool
        True if the files are equal.

    """
    data1 = ascii_handler.collect_data_from_file(file1)
    data2 = ascii_handler.collect_data_from_file(file2)
    data1.pop("schema_version", None)
    data2.pop("schema_version", None)

    _logger.debug(f"Comparing json/yaml files: {file1} and {file2}")

    if data1 == data2:
        return True

    if data1.keys() != data2.keys():
        _logger.error(f"Keys do not match: {data1.keys()} and {data2.keys()}")
        return False
    _comparison = all(
        (
            _compare_value_from_parameter_dict(data1[k], data2[k], tolerance)
            if k == "value"
            else data1[k] == data2[k]
        )
        for k in data1
    )
    if not _comparison:
        _logger.error(f"Values do not match: {data1} and {data2} (tolerance: {tolerance})")
    return _comparison


def _compare_value_from_parameter_dict(data1, data2, tolerance=1.0e-5):
    """Compare value fields given in different formats."""

    def _as_list(value):
        if isinstance(value, str):
            return gen.convert_string_to_list(value)
        if isinstance(value, list | np.ndarray):
            return value
        return [value]

    _logger.info(f"Comparing values: {data1} and {data2} (tolerance: {tolerance})")

    _as_list_1 = _as_list(data1)
    _as_list_2 = _as_list(data2)
    if isinstance(_as_list_1, str):
        return _as_list_1 == _as_list_2
    return np.allclose(_as_list_1, _as_list_2, rtol=tolerance)


def compare_ecsv_files(file1, file2, tolerance=1.0e-5, test_columns=None):
    """
    Compare two ecsv files.

    The comparison is successful if:

    - same number of rows
    - numerical values in columns are close

    The comparison can be restricted to a subset of columns with some additional
    cuts applied. This is configured through the test_columns parameter. This is
    a list of dictionaries, where each dictionary contains the following
    key-value pairs:
    - test_column_name: column name to compare.
    - cut_column_name: column for filtering.
    - cut_condition: condition for filtering.

    Parameters
    ----------
    file1: str
        First file to compare
    file2: str
        Second file to compare
    tolerance: float
        Tolerance for comparing numerical values.
    test_columns: list
        List of columns to compare. If None, all columns are compared.

    """
    _logger.info(f"Comparing files: {file1} and {file2}")
    table1 = Table.read(file1, format="ascii.ecsv")
    table2 = Table.read(file2, format="ascii.ecsv")

    if test_columns is None:
        test_columns = [{"test_column_name": col} for col in table1.colnames]

    def generate_mask(table, column, condition):
        """Generate a boolean mask based on the condition (note the usage of eval)."""
        return (
            eval(f"table['{column}'] {condition}")  # pylint: disable=eval-used
            if condition
            else np.ones(len(table), dtype=bool)
        )

    for col_dict in test_columns:
        col_name = col_dict["test_column_name"]
        mask1 = generate_mask(
            table1, col_dict.get("cut_column_name", ""), col_dict.get("cut_condition", "")
        )
        mask2 = generate_mask(
            table2, col_dict.get("cut_column_name", ""), col_dict.get("cut_condition", "")
        )
        table1_masked, table2_masked = table1[mask1], table2[mask2]

        if len(table1_masked) != len(table2_masked):
            return False

        if np.issubdtype(table1_masked[col_name].dtype, np.floating):
            if not np.allclose(table1_masked[col_name], table2_masked[col_name], rtol=tolerance):
                _logger.warning(f"Column {col_name} outside of relative tolerance {tolerance}")
                return False

    return True


def _validate_simtel_cfg_files(config, simtel_cfg_file):
    """
    Check sim_telarray configuration files and compare with reference file.

    Note the finetuned naming of configuration files by simtools.

    """
    reference_file = Path(simtel_cfg_file)
    test_file = (
        Path(config["configuration"]["output_path"])
        / f"model/{config['configuration']['model_version']}"
        / reference_file.name.replace("_test", f"_{config['configuration']['label']}")
    )
    _logger.info(
        f"Comparing simtel cfg files: {reference_file} and {test_file} "
        f"for model version {config['configuration']['model_version']}"
    )
    assert _compare_simtel_cfg_files(reference_file, test_file)


def _compare_simtel_cfg_files(reference_file, test_file):
    """
    Compare two sim_telarray configuration files.

    Line-by-line string comparison. Requires similar sequence of
    parameters in the files. Ignore lines listed in cfg_ignore_keys
    (e.g., simtools package versions or hadronic interaction model strings).

    Parameters
    ----------
    reference_file: Path
        Reference sim_telarray configuration file.
    test_file: Path
        Test sim_telarray configuration file.

    Returns
    -------
    bool
        True if the files are equal.

    """
    with open(reference_file, encoding="utf-8") as f1, open(test_file, encoding="utf-8") as f2:
        reference_cfg = [line.rstrip() for line in f1 if line.strip()]
        test_cfg = [line.rstrip() for line in f2 if line.strip()]

    def filter_ignored(cfg_lines, file_label):
        filtered = []
        for line in cfg_lines:
            ignored_key = next((ignore for ignore in cfg_ignore_keys if ignore in line), None)
            if ignored_key:
                _logger.debug(f"Ignoring line in {file_label} due to key '{ignored_key}': {line}")
                continue
            filtered.append(line)
        return filtered

    reference_cfg_filtered = filter_ignored(reference_cfg, "reference file")
    test_cfg_filtered = filter_ignored(test_cfg, "test file")

    if len(reference_cfg_filtered) != len(test_cfg_filtered):
        _logger.error(
            f"Line counts differ after filtering: {reference_file} "
            f"({len(reference_cfg_filtered)} lines), "
            f"{test_file} ({len(test_cfg_filtered)} lines)."
        )
        return False

    for ref_line, test_line in zip(reference_cfg_filtered, test_cfg_filtered):
        if ref_line != test_line:
            _logger.error(
                f"Configuration files {reference_file} and {test_file} do not match: "
                f"'{ref_line}' and '{test_line}'"
            )
            return False

    return True

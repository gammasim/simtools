"""Compare application output to reference output."""

import logging
import re
from pathlib import Path

import numpy as np
from astropy.table import Table

import simtools.utils.general as gen
from simtools.db import db_handler
from simtools.io import ascii_handler
from simtools.testing import assertions

_logger = logging.getLogger(__name__)


def validate_application_output(
    config, from_command_line=None, from_config_file=None, db_config=None
):
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

        if from_command_line == from_config_file:
            _validate_output_files(config, integration_test, db_config)

            if "file_type" in integration_test:
                assert assertions.assert_file_type(
                    integration_test["file_type"],
                    Path(config["configuration"]["output_path"]).joinpath(
                        config["configuration"]["output_file"]
                    ),
                )
        _test_simtel_cfg_files(config, integration_test, from_command_line, from_config_file)


def _validate_output_files(config, integration_test, db_config):
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
        _validate_model_parameter_json_file(
            config,
            integration_test["model_parameter_validation"],
            db_config,
        )


def _test_simtel_cfg_files(config, integration_test, from_command_line, from_config_file):
    """Test simtel cfg files."""
    cfg_files = integration_test.get("test_simtel_cfg_files", {})

    source = from_command_line or from_config_file
    sources = source if isinstance(source, list) else [source]

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


def _normalize_model_version(model_version):
    """
    Return model_version as string, or None if list/None (needs special handling).

    Parameters
    ----------
    model_version : str, list, or None
        Model version

    Returns
    -------
    str or None
        String if single version, None otherwise

    """
    if not model_version or isinstance(model_version, list):
        return None
    return model_version


def _extract_version_from_filename(file_str):
    """
    Extract MAJOR.MINOR version from filename or path.

    Parameters
    ----------
    file_str : str
        Filename or path string potentially containing a version

    Returns
    -------
    str or None
        Extracted version string (MAJOR.MINOR) or None if not found

    """
    # Match MAJOR.MINOR (optionally followed by .PATCH) then _ or /
    version_pattern = re.compile(r"(\d{1,10})\.(\d{1,10})(?:\.\d{1,10})?[_/]")
    match = version_pattern.search(file_str)
    if match:
        return f"{match.group(1)}.{match.group(2)}"
    return None


def _try_resolve_single_version_path(base_path, file_str, minor_version):
    """
    Try to resolve path for a single version.

    Parameters
    ----------
    base_path : Path
        Base path to search in
    file_str : str
        File path string
    minor_version : str
        Version in MAJOR.MINOR format

    Returns
    -------
    Path or None
        Resolved path or None if not found

    """
    # Case 1: version as directory
    if "/" in file_str:
        dir_ver, rest = file_str.split("/", 1)
        if minor_version in dir_ver:
            for match in sorted(base_path.glob(f"{minor_version}.*")):
                candidate = match / rest
                if candidate.exists():
                    _logger.debug(f"Resolved {file_str} to {candidate.relative_to(base_path)}")
                    return candidate

    # Case 2: version in filename
    if minor_version in file_str:
        glob_pattern = file_str.replace(minor_version, f"{minor_version}*")
        for match_path in sorted(base_path.glob(glob_pattern)):
            _logger.debug(f"Resolved {file_str} to {match_path.relative_to(base_path)}")
            return match_path

    return None


def _try_resolve_version_path(base_path, file_str, model_version):
    """
    Try to resolve paths with incomplete version info in directory or filename.

    Parameters
    ----------
    base_path : Path
        Base path to search in
    file_str : str
        File path string (may contain version directory or version in filename)
    model_version : str, list, or None
        Model version string, list of version strings, or None if version
        should be extracted from filename

    Examples
    --------
    "6.0/file.md" -> "6.0.2/file.md"
    "file_6.0_name.txt" -> "file_6.0.2_name.txt"
    With list: model_version=["6.0", "6.1"], file "file_6.1_name.txt" -> "file_6.1.5_name.txt"
    """
    if isinstance(model_version, list):
        for version in model_version:
            result = _try_resolve_version_path(base_path, file_str, version)
            if result:
                return result
        return None

    if model_version is None:
        model_version = _extract_version_from_filename(file_str)
        if model_version is None:
            return None

    parts = model_version.split(".")
    if len(parts) < 2:
        return None
    minor_version = f"{parts[0]}.{parts[1]}"

    return _try_resolve_single_version_path(base_path, file_str, minor_version)


def _resolve_output_file_path(output_path, file_str, model_version):
    """
    Resolve output file path, trying version-aware resolution if needed.

    Handles two cases:
    1. Version in directory path: "6.0/file.md" -> "6.0.2/file.md"
    2. Version in filename: "file_6.0_name.txt" -> "file_6.0.2_name.txt"

    Parameters
    ----------
    output_path : str or Path
        Base output path
    file_str : str
        File path string (may contain version directory or version in filename)
    model_version : str
        Model version string

    Returns
    -------
    Path
        Resolved file path

    """
    base_path = Path(output_path)
    output_file_path = base_path / file_str

    if output_file_path.exists():
        return output_file_path

    resolved = _try_resolve_version_path(base_path, file_str, model_version)
    if resolved:
        return resolved

    return output_file_path


def _resolve_file_path_with_versions(output_path, file_str, model_version):
    """
    Resolve file path for single version or list of versions.

    Parameters
    ----------
    output_path : Path or str
        Base output path
    file_str : str
        File path string
    model_version : str, list, or None
        Model version(s)

    Returns
    -------
    Path
        Resolved file path

    """
    versions = (
        model_version
        if isinstance(model_version, list)
        else [_normalize_model_version(model_version)]
    )

    # Try each version until one resolves successfully
    for version in versions:
        try:
            resolved_path = _resolve_output_file_path(output_path, file_str, version)
            if resolved_path.exists():
                return resolved_path
        except (OSError, ValueError):
            continue

    # If none resolved, return direct path
    return Path(output_path) / file_str


def _validate_output_path_and_file(config, integration_file_tests):
    """Check if output paths and files exist."""
    for file_test in integration_file_tests:
        try:
            output_path = config["configuration"][file_test["path_descriptor"]]
        except KeyError as exc:
            raise KeyError(
                f"Path {file_test['path_descriptor']} not found in integration test configuration."
            ) from exc

        model_version = config["configuration"].get("model_version")
        output_file_path = _resolve_file_path_with_versions(
            output_path, file_test["file"], model_version
        )

        _logger.info(f"Checking path: {output_file_path}")
        try:
            assert output_file_path.exists()
        except AssertionError as exc:
            raise AssertionError(f"Output file {output_file_path} does not exist. ") from exc

        assert assertions.check_output_from_sim_telarray(output_file_path, file_test)


def _validate_model_parameter_json_file(config, model_parameter_validation, db_config):
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
    db = db_handler.DatabaseHandler(db_config=db_config)

    reference_parameter_name = model_parameter_validation.get("reference_parameter_name")

    # Construct parameter file path to extract version from it
    parameter_file = (
        Path(config["configuration"]["output_path"])
        / config["configuration"].get("telescope")
        / model_parameter_validation["parameter_file"]
    )

    # Try to extract version from the parameter file path
    model_version = _extract_version_from_filename(str(parameter_file))

    # If no version found in path, fall back to config model_version
    if model_version is None:
        raw_model_version = config["configuration"].get("model_version")
        model_version = (
            raw_model_version[0]
            if isinstance(raw_model_version, list) and raw_model_version
            else _normalize_model_version(raw_model_version)
        )

    reference_model_parameter = db.get_model_parameter(
        parameter=reference_parameter_name,
        site=config["configuration"].get("site"),
        array_element_name=config["configuration"].get("telescope"),
        model_version=model_version,
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

    File names with version patterns are resolved accordingly (first tries
    MAJOR.MINOR.PATCH, then falls back to MAJOR.MINOR).

    Note the finetuned naming of configuration files by simtools.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    simtel_cfg_file : str
        Reference sim_telarray configuration file.

    """
    reference_file = Path(simtel_cfg_file)
    output_path = Path(config["configuration"]["output_path"])

    expected_filename = reference_file.name.replace("_test", f"_{config['configuration']['label']}")

    # Extract version from reference file name (e.g., "CTA-South-LSTS-01-6.0_test.cfg" -> "6.0")
    model_version = _extract_version_from_filename(reference_file.name)

    test_file = _resolve_output_file_path(
        output_path / "model", f"{model_version}/{expected_filename}", model_version
    )

    _logger.info(
        f"Comparing simtel cfg files: {reference_file} and {test_file} "
        f"for model version {model_version}"
    )
    assert _compare_simtel_cfg_files(reference_file, test_file)


def _lines_match_with_version_flexibility(ref_line, test_line):
    """
    Compare two lines with flexibility for version differences.

    Strategy:
    - Exact string equality passes.
    - Otherwise, normalize all version patterns in both lines to MAJOR.MINOR and compare.

    Parameters
    ----------
    ref_line : str
        Reference line
    test_line : str
        Test line

    Returns
    -------
    bool
        True if lines match (considering version flexibility)

    """
    if ref_line == test_line:
        return True

    # Match MAJOR.MINOR or MAJOR.MINOR.PATCH even when followed by underscores/letters
    # Ensure we don't match when adjacent to other digits
    version_re = re.compile(r"(?<!\d)(\d{1,10})\.(\d{1,10})(?:\.(\d{1,10}))?(?!\d)")

    def _to_minor(match: re.Match) -> str:
        return f"{match.group(1)}.{match.group(2)}"

    ref_norm = version_re.sub(_to_minor, ref_line)
    test_norm = version_re.sub(_to_minor, test_line)

    if ref_norm == test_norm:
        if ref_line != test_line:
            _logger.debug(f"Lines match after version normalization: '{ref_line}' vs '{test_line}'")
        return True

    return False


def _compare_simtel_cfg_files(reference_file, test_file):
    """
    Compare two sim_telarray configuration files.

    Line-by-line string comparison with version flexibility. Requires similar sequence of
    parameters in the files. Ignore lines containing 'config_release', 'Label', or
    'simtools_version'. For all version patterns (X.Y or X.Y.Z), compare MAJOR.MINOR only,
    allowing reference and test files to differ in patch versions.

    Parameters
    ----------
    reference_file: Path
        Reference sim_telarray configuration file.
    test_file: Path
        Test sim_telarray configuration file.

    Returns
    -------
    bool
        True if the files are equal (considering version flexibility).

    """
    with open(reference_file, encoding="utf-8") as f1, open(test_file, encoding="utf-8") as f2:
        reference_cfg = [line.rstrip() for line in f1 if line.strip()]
        test_cfg = [line.rstrip() for line in f2 if line.strip()]

    if len(reference_cfg) != len(test_cfg):
        _logger.error(
            f"Line counts differ: {reference_file} ({len(reference_cfg)} lines), "
            f"{test_file} ({len(test_cfg)} lines)."
        )
        return False

    for ref_line, test_line in zip(reference_cfg, test_cfg):
        if any(ignore in ref_line for ignore in ("config_release", "Label", "simtools_version")):
            continue

        if not _lines_match_with_version_flexibility(ref_line, test_line):
            _logger.error(
                f"Configuration files {reference_file} and {test_file} do not match: "
                f"'{ref_line}' and '{test_line}'"
            )
            return False

    return True

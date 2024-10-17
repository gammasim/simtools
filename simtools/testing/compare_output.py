"""Compare application output to reference output."""

import logging
from pathlib import Path

import numpy as np
from astropy.table import Table

import simtools.utils.general as gen

_logger = logging.getLogger(__name__)


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
        True if the files are equal, False otherwise.

    """
    _file1_suffix = Path(file1).suffix
    _file2_suffix = Path(file2).suffix
    if _file1_suffix != _file2_suffix:
        raise ValueError(f"File suffixes do not match: {file1} and {file2}")
    if _file1_suffix == ".ecsv":
        return compare_ecsv_files(file1, file2, tolerance, test_columns)
    if _file1_suffix in (".json", ".yaml", ".yml"):
        return compare_json_or_yaml_files(file1, file2)

    _logger.warning(f"Unknown file type for files: {file1} and {file2}")
    return False


def compare_json_or_yaml_files(file1, file2, tolerance=1.0e-2):
    """
    Compare two json or yaml files.

    Take into account float comparison for sim_telarray string-embedded floats.

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
        True if the files are equal, False otherwise.

    """
    data1 = gen.collect_data_from_file_or_dict(file1, in_dict=None)
    data2 = gen.collect_data_from_file_or_dict(file2, in_dict=None)

    _logger.debug(f"Comparing json/yaml files: {file1} and {file2}")

    if data1 == data2:
        return True

    if "value" in data1 and isinstance(data1["value"], str):
        value_list_1 = gen.convert_string_to_list(data1.pop("value"))
        value_list_2 = gen.convert_string_to_list(data2.pop("value"))
        return np.allclose(value_list_1, value_list_2, rtol=tolerance)
    return data1 == data2


def compare_ecsv_files(file1, file2, tolerance=1.0e-5, test_columns=None):
    """
    Compare two ecsv files.

    The comparison is successful if:

    - same number of rows
    - numerical values in columns are close

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

    comparison_result = len(table1) == len(table2)

    test_columns = test_columns if test_columns else table1.colnames

    for col_name in test_columns:
        if np.issubdtype(table1[col_name].dtype, np.floating):
            comparison_result = comparison_result and np.allclose(
                table1[col_name], table2[col_name], rtol=tolerance
            )

    return comparison_result

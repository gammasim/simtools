import numpy as np
import pytest
from astropy.table import Table

from simtools.io_operations.io_table_handler import (
    _merge,
    merge_tables,
    read_table_file_type,
    read_tables,
    write_tables,
)

# Constants to avoid string duplication
H5PY_IMPORT = "h5py is required"
TABLE_HANDLER_PATH = "simtools.io_operations.io_table_handler"
TEST_TABLE_NAME = "test_table"


# Common fixtures for mocked dependencies
@pytest.fixture
def mock_table():
    """Create a mock table with test data."""
    table = Table({"col1": [1, 2]})
    table.meta["EXTNAME"] = "test_table"
    return table


@pytest.fixture
def mock_read_type(mocker):
    """Mock read_table_file_type."""
    return mocker.patch("simtools.io_operations.io_table_handler.read_table_file_type")


@pytest.fixture
def mock_table_read(mocker):
    """Mock Table.read."""
    return mocker.patch("astropy.table.Table.read")


@pytest.fixture
def mock_table_write(mocker):
    """Mock Table.write."""
    return mocker.patch("astropy.table.Table.write")


@pytest.fixture
def mock_fits_objects(mocker):
    """Mock FITS objects and functions."""
    return {
        "primary_hdu": mocker.patch("astropy.io.fits.PrimaryHDU"),
        "table_hdu": mocker.patch("astropy.io.fits.BinTableHDU"),
        "hdul": mocker.patch("astropy.io.fits.HDUList"),
    }


@pytest.fixture
def mock_logger(mocker):
    """Mock logger."""
    return mocker.patch("simtools.io_operations.io_table_handler._logger")


def test_read_table_file_type_empty_list():
    with pytest.raises(ValueError, match="No input files provided."):
        read_table_file_type([])


def test_read_table_file_type_all_fits():
    input_files = ["test.fits", "test2.fits.gz"]
    assert read_table_file_type(input_files) == "FITS"


def test_read_table_file_type_all_hdf5():
    input_files = ["test.hdf5", "test2.h5"]
    assert read_table_file_type(input_files) == "HDF5"


def test_read_table_file_type_mixed_types():
    input_files = ["test.fits", "test.hdf5"]
    with pytest.raises(ValueError, match="All input files must be of the same type"):
        read_table_file_type(input_files)


def test_read_table_file_type_unsupported():
    input_files = ["test.txt"]
    with pytest.raises(ValueError, match="Unsupported file type"):
        read_table_file_type(input_files)


def test_read_table_file_type_missing_h5py(mocker):
    mocker.patch("importlib.util.find_spec", return_value=None)
    input_files = ["test.hdf5"]
    with pytest.raises(ImportError, match=H5PY_IMPORT):
        read_table_file_type(input_files)


def test_merge_single_table(mocker):
    # Mock read_tables function
    mock_read = mocker.patch(f"{TABLE_HANDLER_PATH}.read_tables")

    # Create test tables
    table1 = Table({"col1": [1, 2], "file_id": [0, 0]})
    table2 = Table({"col1": [3, 4], "file_id": [0, 0]})

    mock_read.side_effect = [
        {TEST_TABLE_NAME: table1},
        {TEST_TABLE_NAME: table2},
    ]

    result = _merge(["file1.fits", "file2.fits"], ["test_table"], "FITS")

    assert len(result) == 1
    assert "test_table" in result
    assert len(result["test_table"]) == 4
    assert np.array_equal(result["test_table"]["file_id"], [0, 0, 1, 1])


def test_merge_multiple_tables(mocker):
    # Mock read_tables function
    mock_read = mocker.patch("simtools.io_operations.io_table_handler.read_tables")

    # Create test tables
    tables1 = {
        "table1": Table({"col1": [1], "file_id": [0]}),
        "table2": Table({"col2": [2], "file_id": [0]}),
    }
    tables2 = {
        "table1": Table({"col1": [3], "file_id": [0]}),
        "table2": Table({"col2": [4], "file_id": [0]}),
    }

    mock_read.side_effect = [tables1, tables2]

    result = _merge(["file1.fits", "file2.fits"], ["table1", "table2"], "FITS")

    assert len(result) == 2
    assert all(name in result for name in ["table1", "table2"])
    assert len(result["table1"]) == 2
    assert len(result["table2"]) == 2


def test_merge_without_file_id(mocker):
    # Mock read_tables function
    mock_read = mocker.patch("simtools.io_operations.io_table_handler.read_tables")

    # Create test tables without file_id
    table1 = Table({"col1": [1, 2]})
    table2 = Table({"col1": [3, 4]})

    mock_read.side_effect = [{"test_table": table1}, {"test_table": table2}]

    result = _merge(["file1.fits", "file2.fits"], ["test_table"], "FITS")

    assert len(result) == 1
    assert "test_table" in result
    assert len(result["test_table"]) == 4
    assert "file_id" not in result["test_table"].colnames


def test_read_tables_fits(mocker):
    # Mock Table.read
    mock_read = mocker.patch("astropy.table.Table.read")
    mock_table = Table({"col1": [1, 2]})
    mock_read.return_value = mock_table

    # Mock read_table_file_type
    mock_file_type = mocker.patch("simtools.io_operations.io_table_handler.read_table_file_type")
    mock_file_type.return_value = "FITS"

    result = read_tables("test.fits", ["table1", "table2"])

    assert len(result) == 2
    assert all(name in result for name in ["table1", "table2"])
    assert mock_read.call_count == 2
    mock_read.assert_has_calls(
        [mocker.call("test.fits", hdu="table1"), mocker.call("test.fits", hdu="table2")]
    )


def test_read_tables_hdf5(mocker):
    # Mock Table.read
    mock_read = mocker.patch("astropy.table.Table.read")
    mock_table = Table({"col1": [1, 2]})
    mock_read.return_value = mock_table

    # Mock read_table_file_type
    mock_file_type = mocker.patch("simtools.io_operations.io_table_handler.read_table_file_type")
    mock_file_type.return_value = "HDF5"

    result = read_tables("test.h5", ["table1", "table2"])

    assert len(result) == 2
    assert all(name in result for name in ["table1", "table2"])
    assert mock_read.call_count == 2
    mock_read.assert_has_calls(
        [mocker.call("test.h5", path="table1"), mocker.call("test.h5", path="table2")]
    )


def test_read_tables_unsupported_format(mocker):
    # Mock read_table_file_type
    mock_file_type = mocker.patch("simtools.io_operations.io_table_handler.read_table_file_type")
    mock_file_type.return_value = "CSV"

    with pytest.raises(ValueError, match="Unsupported file format"):
        read_tables("test.csv", ["table1"])


def test_read_tables_explicit_file_type(mocker):
    # Mock Table.read
    mock_read = mocker.patch("astropy.table.Table.read")
    mock_table = Table({"col1": [1, 2]})
    mock_read.return_value = mock_table

    # Mock read_table_file_type to ensure it's not called
    mock_file_type = mocker.patch("simtools.io_operations.io_table_handler.read_table_file_type")

    result = read_tables("test.fits", ["table1"], file_type="FITS")

    assert len(result) == 1
    mock_file_type.assert_not_called()
    mock_read.assert_called_once_with("test.fits", hdu="table1")


def test_write_tables_fits(tmp_path, mock_table, mock_fits_objects):
    """Test writing tables in FITS format."""
    output_file = tmp_path / "test.fits"
    write_tables([mock_table], output_file, file_type="FITS")

    mock_fits_objects["hdul"].assert_called_once()
    mock_fits_objects["hdul"].return_value.writeto.assert_called_once_with(
        output_file, checksum=False
    )


def test_write_tables_hdf5(tmp_path, mock_table, mock_table_write):
    output_file = tmp_path / "test.h5"
    write_tables([mock_table], output_file, file_type="HDF5")

    mock_table_write.assert_called_once_with(
        output_file,
        path="test_table",
        append=True,
        format="hdf5",
        serialize_meta=True,
        compression=True,
    )


def test_write_tables_dict_input(tmp_path, mocker):
    mock_table = Table({"col1": [1, 2]})
    mock_table.meta["EXTNAME"] = "test_table"
    tables_dict = {"table1": mock_table}

    mock_write = mocker.patch("astropy.table.Table.write")

    output_file = tmp_path / "test.h5"

    write_tables(tables_dict, output_file, file_type="HDF5")

    mock_write.assert_called_once()


def test_write_tables_existing_file(tmp_path, mocker):
    """Test writing tables when output file exists."""
    mock_table = Table({"col1": [1, 2]})
    mock_table.meta["EXTNAME"] = "test_table"

    output_file = tmp_path / "test.fits"
    output_file.touch()  # Create the file

    # Mock necessary FITS objects and functions
    mocker.patch("astropy.io.fits.PrimaryHDU")
    mocker.patch("astropy.io.fits.BinTableHDU")
    mock_hdul = mocker.patch("astropy.io.fits.HDUList")

    write_tables([mock_table], output_file, file_type="FITS")

    assert not output_file.exists()  # File should be deleted before writing
    mock_hdul.assert_called_once()


def test_write_tables_no_file_type(tmp_path, mock_table, mock_read_type, mock_fits_objects):
    """Test writing tables without explicit file type."""
    mock_read_type.return_value = "FITS"
    output_file = tmp_path / "test.fits"

    write_tables([mock_table], output_file)

    mock_read_type.assert_called_once_with([output_file])
    mock_fits_objects["hdul"].assert_called_once()


def test_merge_tables_success(mock_read_type, mock_logger, mocker):
    """Test successful table merging."""
    mock_merge = mocker.patch("simtools.io_operations.io_table_handler._merge")
    mock_write = mocker.patch("simtools.io_operations.io_table_handler.write_tables")

    input_files = ["file1.fits", "file2.fits"]
    table_names = ["table1", "table2"]
    output_file = "output.fits"

    mock_read_type.return_value = "FITS"
    mock_merge.return_value = {"table1": mocker.Mock(), "table2": mocker.Mock()}

    merge_tables(input_files, table_names, output_file)

    mock_read_type.assert_called_once_with(input_files)
    mock_merge.assert_called_once_with(input_files, table_names, "FITS")
    mock_write.assert_called_once_with(mock_merge.return_value, output_file, "FITS")
    mock_logger.info.assert_called_once_with("Merging 2 files into output.fits")


def test_merge_tables_hdf5(mocker):
    # Mock dependencies
    mock_read_type = mocker.patch("simtools.io_operations.io_table_handler.read_table_file_type")
    mock_merge = mocker.patch("simtools.io_operations.io_table_handler._merge")
    mock_write = mocker.patch("simtools.io_operations.io_table_handler.write_tables")
    mocker.patch("simtools.io_operations.io_table_handler._logger")

    # Test data
    input_files = ["file1.hdf5", "file2.hdf5"]
    table_names = ["table1"]
    output_file = "output.hdf5"

    # Configure mocks
    mock_read_type.return_value = "HDF5"
    mock_merge.return_value = {"table1": mocker.Mock()}

    # Execute function
    merge_tables(input_files, table_names, output_file)

    # Verify calls
    mock_read_type.assert_called_once_with(input_files)
    mock_merge.assert_called_once_with(input_files, table_names, "HDF5")
    mock_write.assert_called_once_with(mock_merge.return_value, output_file, "HDF5")


def test_merge_tables_propagates_errors(mocker):
    # Mock dependencies to raise errors
    mock_read_type = mocker.patch("simtools.io_operations.io_table_handler.read_table_file_type")
    mock_read_type.side_effect = ValueError("Test error")

    # Test data
    input_files = ["file1.txt"]
    table_names = ["table1"]
    output_file = "output.fits"

    # Verify error propagation
    with pytest.raises(ValueError, match="Test error"):
        merge_tables(input_files, table_names, output_file)

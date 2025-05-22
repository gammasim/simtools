import numpy as np
import pytest
from astropy.table import Table

from simtools.io_operations.io_table_handler import (
    _merge,
    copy_metadata_to_hdf5,
    merge_tables,
    read_table_file_type,
    read_tables,
    write_table_in_hdf5,
    write_tables,
)

# Constants for repeated strings
TABLE_HANDLER_PATH = "simtools.io_operations.io_table_handler"
READ_TABLE_FILE_TYPE = f"{TABLE_HANDLER_PATH}.read_table_file_type"
ASTROPY_TABLE_READ = "astropy.table.Table.read"
H5PY_FILE = "h5py.File"
TEST_TABLE_NAME = "test_table"
FILE_ID = "file_id"

# Test file paths
TEST_FITS = "test.fits"
TEST_HDF5 = "test.hdf5"
TEST_H5 = "test.h5"
OUTPUT_FITS = "output.fits"
FILE1_FITS = "file1.fits"
FILE2_FITS = "file2.fits"
SOURCE_H5 = "source.h5"
DEST_H5 = "dest.h5"


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


@pytest.fixture
def mock_h5py_file(mocker):
    """Mock h5py.File context manager."""
    mock_file = mocker.MagicMock()
    mock_context = mocker.MagicMock()
    mock_context.__enter__.return_value = mock_file
    mocker.patch("h5py.File", return_value=mock_context)
    return mock_file


def test_read_table_file_type_empty_list():
    with pytest.raises(ValueError, match="No input files provided."):
        read_table_file_type([])


def test_read_table_file_type_all_fits():
    input_files = [TEST_FITS, "test2.fits.gz"]
    assert read_table_file_type(input_files) == "FITS"


def test_read_table_file_type_all_hdf5():
    input_files = [TEST_HDF5, TEST_H5]
    assert read_table_file_type(input_files) == "HDF5"


def test_read_table_file_type_mixed_types():
    input_files = [TEST_FITS, TEST_HDF5]
    with pytest.raises(ValueError, match="All input files must be of the same type"):
        read_table_file_type(input_files)


def test_read_table_file_type_unsupported():
    input_files = ["test.txt"]
    with pytest.raises(ValueError, match="Unsupported file type"):
        read_table_file_type(input_files)


def test_read_table_file_type_missing_h5py(mocker):
    mocker.patch("importlib.util.find_spec", return_value=None)
    input_files = [TEST_HDF5]
    with pytest.raises(ImportError, match="h5py is required"):
        read_table_file_type(input_files)


def test_merge_single_table(mocker, tmp_path):
    """Test merging single table from multiple files."""
    # Mock read_tables function
    mock_read = mocker.patch(f"{TABLE_HANDLER_PATH}.read_tables")

    # Create test tables
    table1 = Table({"col1": [1, 2], FILE_ID: [0, 0]})
    table2 = Table({"col1": [3, 4], FILE_ID: [0, 0]})

    mock_read.side_effect = [
        {TEST_TABLE_NAME: table1},
        {TEST_TABLE_NAME: table2},
    ]

    output_file = tmp_path / "output.fits"
    result = _merge(["file1.fits", "file2.fits"], [TEST_TABLE_NAME], "FITS", output_file)

    assert len(result) == 1
    assert TEST_TABLE_NAME in result
    assert len(result[TEST_TABLE_NAME]) == 4
    assert np.array_equal(result[TEST_TABLE_NAME][FILE_ID], [0, 0, 1, 1])


def test_merge_multiple_tables(mocker, tmp_path):
    """Test merging multiple tables from multiple files."""
    mock_read = mocker.patch(f"{TABLE_HANDLER_PATH}.read_tables")
    tables1 = {
        "table1": Table({"col1": [1], FILE_ID: [0]}),
        "table2": Table({"col2": [2], FILE_ID: [0]}),
    }
    tables2 = {
        "table1": Table({"col1": [3], FILE_ID: [0]}),
        "table2": Table({"col2": [4], FILE_ID: [0]}),
    }
    mock_read.side_effect = [tables1, tables2]

    output_file = tmp_path / "output.fits"
    result = _merge(["file1.fits", "file2.fits"], ["table1", "table2"], "FITS", output_file)

    assert len(result) == 2
    assert all(name in result for name in ["table1", "table2"])
    assert len(result["table1"]) == 2
    assert len(result["table2"]) == 2


def test_merge_without_file_id(mocker, tmp_path):
    """Test merging tables without file_id column."""
    mock_read = mocker.patch(f"{TABLE_HANDLER_PATH}.read_tables")
    table1 = Table({"col1": [1, 2]})
    table2 = Table({"col1": [3, 4]})
    mock_read.side_effect = [{TEST_TABLE_NAME: table1}, {TEST_TABLE_NAME: table2}]

    output_file = tmp_path / "output.fits"
    result = _merge(["file1.fits", "file2.fits"], [TEST_TABLE_NAME], "FITS", output_file)

    assert len(result) == 1
    assert TEST_TABLE_NAME in result
    assert len(result[TEST_TABLE_NAME]) == 4
    assert FILE_ID not in result[TEST_TABLE_NAME].colnames


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


def test_write_tables_hdf5(tmp_path, mock_table, mock_h5py_file):
    """Test writing tables in HDF5 format."""
    output_file = tmp_path / "test.h5"
    write_tables([mock_table], output_file, file_type="HDF5")

    # Verify h5py operations were called correctly
    mock_h5py_file.create_dataset.assert_called_once()
    call_args = mock_h5py_file.create_dataset.call_args[1]
    assert call_args["compression"] == "gzip"
    assert call_args["compression_opts"] == 4


def test_write_tables_dict_input(tmp_path, mock_table, mock_h5py_file):
    """Test writing dictionary of tables."""
    tables_dict = {"table1": mock_table}
    output_file = tmp_path / "test.h5"
    write_tables(tables_dict, output_file, file_type="HDF5")

    # Verify h5py operations were called
    mock_h5py_file.create_dataset.assert_called_once()


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
    mock_merge = mocker.patch(f"{TABLE_HANDLER_PATH}._merge")
    mock_write = mocker.patch(f"{TABLE_HANDLER_PATH}.write_tables")

    input_files = ["file1.fits", "file2.fits"]
    table_names = ["table1", "table2"]
    output_file = "output.fits"

    mock_read_type.return_value = "FITS"
    mock_merge.return_value = {"table1": mocker.Mock(), "table2": mocker.Mock()}

    merge_tables(input_files, table_names, output_file)

    mock_read_type.assert_called_once_with(input_files)
    mock_merge.assert_called_once_with(input_files, table_names, "FITS", output_file)
    mock_write.assert_called_once_with(mock_merge.return_value, output_file, "FITS")


def test_merge_tables_hdf5(mocker):
    """Test merging with HDF5 files."""
    mock_read_type = mocker.patch(f"{TABLE_HANDLER_PATH}.read_table_file_type")
    mock_merge = mocker.patch(f"{TABLE_HANDLER_PATH}._merge")
    mocker.patch(f"{TABLE_HANDLER_PATH}.write_tables")

    input_files = ["file1.hdf5", "file2.hdf5"]
    table_names = ["table1"]
    output_file = "output.hdf5"

    mock_read_type.return_value = "HDF5"
    mock_merge.return_value = {"table1": mocker.Mock()}

    merge_tables(input_files, table_names, output_file)

    mock_read_type.assert_called_once_with(input_files)
    mock_merge.assert_called_once_with(input_files, table_names, "HDF5", output_file)


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


def test_copy_metadata_to_hdf5(mocker):
    """Test copying metadata between HDF5 files."""
    # Create mock source and destination files
    mock_h5py = mocker.patch("h5py.File")
    mock_src = mocker.MagicMock()
    mock_dst = mocker.MagicMock()

    # Configure File context manager to return our mocks
    mock_h5py.return_value.__enter__.side_effect = [mock_src, mock_dst]

    # Set up source file mock with metadata
    table_name = "test_table"
    meta_name = f"{table_name}.__table_column_meta__"
    mock_src.__contains__.return_value = True
    mock_dst.__contains__.return_value = False

    # Test copying metadata
    copy_metadata_to_hdf5("source.h5", "dest.h5", table_name)

    # Verify file opening modes
    mock_h5py.assert_any_call("source.h5", "r")
    mock_h5py.assert_any_call("dest.h5", "a")

    # Verify metadata was copied
    mock_src.copy.assert_called_once_with(meta_name, mock_dst, name=meta_name)


def test_copy_metadata_to_hdf5_overwrite(mock_h5py_file, mocker):
    """Test copying metadata when it already exists in destination."""
    mock_src = mocker.MagicMock()
    mock_dst = mocker.MagicMock()

    mock_context = mocker.MagicMock()
    mock_context.__enter__.side_effect = [mock_src, mock_dst]
    mocker.patch("h5py.File", return_value=mock_context)

    table_name = "test_table"
    meta_name = f"{table_name}.__table_column_meta__"

    # Configure mocks to indicate metadata exists in both files
    mock_src.__contains__.return_value = True
    mock_dst.__contains__.return_value = True

    copy_metadata_to_hdf5("source.h5", "dest.h5", table_name)

    # Verify existing metadata was deleted
    mock_dst.__delitem__.assert_called_once_with(meta_name)

    # Verify new metadata was copied
    mock_src.copy.assert_called_once_with(meta_name, mock_dst, name=meta_name)


def test_copy_metadata_to_hdf5_no_metadata(mock_h5py_file, mocker):
    """Test when source file has no metadata to copy."""
    mock_src = mocker.MagicMock()
    mock_dst = mocker.MagicMock()

    mock_context = mocker.MagicMock()
    mock_context.__enter__.side_effect = [mock_src, mock_dst]
    mocker.patch("h5py.File", return_value=mock_context)

    table_name = "test_table"

    # Configure mock to indicate no metadata in source
    mock_src.__contains__.return_value = False

    copy_metadata_to_hdf5("source.h5", "dest.h5", table_name)

    # Verify no copy operation was performed
    mock_src.copy.assert_not_called()
    mock_dst.__delitem__.assert_not_called()


def test_write_table_in_hdf5_new_table(mocker, mock_h5py_file, mock_table):
    """Test writing a new table to HDF5 file."""
    mock_h5py_file.__contains__.return_value = False
    mock_dataset = mocker.MagicMock()
    mock_h5py_file.create_dataset.return_value = mock_dataset

    write_table_in_hdf5(mock_table, "test.h5", "test_table")

    # Verify dataset creation
    mock_h5py_file.create_dataset.assert_called_once()
    call_args = mock_h5py_file.create_dataset.call_args[1]
    assert call_args["compression"] == "gzip"
    assert call_args["compression_opts"] == 4
    assert call_args["chunks"] is True

    # Verify metadata was written
    assert mock_dataset.attrs.__setitem__.call_count == len(mock_table.meta)
    mock_dataset.attrs.__setitem__.assert_any_call("EXTNAME", "test_table")


def test_write_table_in_hdf5_append(mock_h5py_file, mock_table):
    """Test appending to existing table in HDF5 file."""
    mock_h5py_file.__contains__.return_value = True
    mock_dataset = mock_h5py_file.__getitem__.return_value
    mock_dataset.shape = (2, 2)  # Initial shape

    write_table_in_hdf5(mock_table, "test.h5", "test_table")

    # Verify dataset was resized and data appended
    mock_dataset.resize.assert_called_once()
    mock_dataset.__setitem__.assert_called_once()


def test_write_table_in_hdf5_empty_table(mock_h5py_file):
    """Test writing empty table to HDF5 file."""
    empty_table = Table()
    empty_table.meta["EXTNAME"] = "empty_table"

    write_table_in_hdf5(empty_table, "test.h5", "empty_table")

    # Verify dataset creation was called with empty data
    mock_h5py_file.create_dataset.assert_called_once()
    call_args = mock_h5py_file.create_dataset.call_args[1]
    assert np.array_equal(call_args["data"], np.array([]))


def test_merge_hdf5_tables(tmp_path, mock_h5py_file, mocker):
    """Test _merge function with HDF5 file type."""
    # Mock read_tables function
    mock_read = mocker.patch("simtools.io_operations.io_table_handler.read_tables")
    mock_write = mocker.patch("simtools.io_operations.io_table_handler.write_table_in_hdf5")
    mock_copy = mocker.patch("simtools.io_operations.io_table_handler.copy_metadata_to_hdf5")

    # Create test tables
    table1 = Table({"col1": [1, 2], "file_id": [0, 0]})
    table2 = Table({"col1": [3, 4], "file_id": [0, 0]})
    table1.meta["EXTNAME"] = "test_table"
    table2.meta["EXTNAME"] = "test_table"

    mock_read.side_effect = [{"test_table": table1}, {"test_table": table2}]

    output_file = tmp_path / "output.h5"
    result = _merge(["file1.h5", "file2.h5"], ["test_table"], "HDF5", output_file)

    # Verify write_table_in_hdf5 was called for each table
    assert mock_write.call_count == 2
    mock_write.assert_has_calls(
        [
            mocker.call(table1, output_file, "test_table"),
            mocker.call(table2, output_file, "test_table"),
        ]
    )

    # Verify copy_metadata was called only for first file
    mock_copy.assert_called_once_with("file1.h5", output_file, "test_table")

    # Verify file_ids were updated
    assert np.array_equal(mock_write.call_args_list[0][0][0]["file_id"], [0, 0])
    assert np.array_equal(mock_write.call_args_list[1][0][0]["file_id"], [1, 1])

    # For HDF5, result should be empty since tables are written directly
    assert result == {"test_table": []}


def test_merge_hdf5_multiple_tables(tmp_path, mock_h5py_file, mocker):
    """Test _merge with multiple HDF5 tables."""
    mock_read = mocker.patch("simtools.io_operations.io_table_handler.read_tables")
    mock_write = mocker.patch("simtools.io_operations.io_table_handler.write_table_in_hdf5")
    mock_copy = mocker.patch("simtools.io_operations.io_table_handler.copy_metadata_to_hdf5")

    tables1 = {
        "table1": Table({"col1": [1], "file_id": [0]}),
        "table2": Table({"col2": [2], "file_id": [0]}),
    }
    tables2 = {
        "table1": Table({"col1": [3], "file_id": [0]}),
        "table2": Table({"table2": [4], "file_id": [0]}),
    }

    tables1["table1"].meta["EXTNAME"] = "table1"
    tables1["table2"].meta["EXTNAME"] = "table2"
    tables2["table1"].meta["EXTNAME"] = "table1"
    tables2["table2"].meta["EXTNAME"] = "table2"

    mock_read.side_effect = [tables1, tables2]

    output_file = tmp_path / "output.h5"
    result = _merge(["file1.h5", "file2.h5"], ["table1", "table2"], "HDF5", output_file)

    # Verify write_table_in_hdf5 was called for each table
    assert mock_write.call_count == 4

    # Verify metadata copy was called once per table
    assert mock_copy.call_count == 2
    mock_copy.assert_has_calls(
        [
            mocker.call("file1.h5", output_file, "table1"),
            mocker.call("file1.h5", output_file, "table2"),
        ]
    )

    # For HDF5, result should contain empty lists
    assert result == {"table1": [], "table2": []}

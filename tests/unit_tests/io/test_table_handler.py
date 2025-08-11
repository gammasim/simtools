import astropy.units as u
import h5py
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from simtools.io.table_handler import (
    _merge,
    _read_table_list_fits,
    _read_table_list_hdf5,
    copy_metadata_to_hdf5,
    merge_tables,
    read_table_file_type,
    read_table_from_hdf5,
    read_table_list,
    read_tables,
    write_table_in_hdf5,
    write_tables,
)

# Constants for repeated strings
TABLE_HANDLER_PATH = "simtools.io.table_handler"
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
FILE1_H5 = "file1.h5"
FILE2_H5 = "file2.h5"
OUTPUT_HDF5 = "output.hdf5"
TEST_CSV = "test.csv"


@pytest.fixture
def mock_table():
    """Create a mock table with test data."""
    table = Table({"col1": [1, 2]})
    table.meta["EXTNAME"] = TEST_TABLE_NAME
    return table


@pytest.fixture
def mock_read_type(mocker):
    """Mock read_table_file_type."""
    return mocker.patch(READ_TABLE_FILE_TYPE)


@pytest.fixture
def mock_table_read(mocker):
    """Mock Table.read."""
    return mocker.patch(ASTROPY_TABLE_READ)


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
    return mocker.patch("simtools.io.table_handler._logger")


@pytest.fixture
def mock_h5py_file(mocker):
    """Mock h5py.File context manager."""
    mock_file = mocker.MagicMock()
    mock_context = mocker.MagicMock()
    mock_context.__enter__.return_value = mock_file
    mocker.patch(H5PY_FILE, return_value=mock_context)
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

    output_file = tmp_path / OUTPUT_FITS
    result = _merge([FILE1_FITS, FILE2_FITS], [TEST_TABLE_NAME], "FITS", output_file)

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

    output_file = tmp_path / OUTPUT_FITS
    result = _merge([FILE1_FITS, FILE2_FITS], ["table1", "table2"], "FITS", output_file)

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

    output_file = tmp_path / OUTPUT_FITS
    result = _merge([FILE1_FITS, FILE2_FITS], [TEST_TABLE_NAME], "FITS", output_file)

    assert len(result) == 1
    assert TEST_TABLE_NAME in result
    assert len(result[TEST_TABLE_NAME]) == 4
    assert FILE_ID not in result[TEST_TABLE_NAME].colnames


def test_merge_hdf5(mocker, tmp_path):
    mock_read = mocker.patch(f"{TABLE_HANDLER_PATH}.read_tables")
    mock_copy = mocker.patch(f"{TABLE_HANDLER_PATH}.copy_metadata_to_hdf5")
    table1 = Table({"col1": [1, 2]})
    table2 = Table({"col1": [3, 4]})
    mock_read.side_effect = [{TEST_TABLE_NAME: table1}, {TEST_TABLE_NAME: table2}]

    output_file = tmp_path / OUTPUT_HDF5
    result = _merge([FILE1_H5, FILE2_H5], [TEST_TABLE_NAME], "HDF5", output_file)

    assert len(result) == 1
    assert mock_copy.call_count == 1

    # Mock Table.read
    mock_read = mocker.patch(ASTROPY_TABLE_READ)
    mock_table = Table({"col1": [1, 2]})
    mock_read.return_value = mock_table

    mock_file_type = mocker.patch(READ_TABLE_FILE_TYPE)
    mock_file_type.return_value = "FITS"

    result = read_tables(TEST_FITS, ["table1", "table2"])

    assert len(result) == 2
    assert all(name in result for name in ["table1", "table2"])
    assert mock_read.call_count == 2
    mock_read.assert_has_calls(
        [mocker.call(TEST_FITS, hdu="table1"), mocker.call(TEST_FITS, hdu="table2")]
    )


def test_read_tables_hdf5(mocker):
    """Test reading tables from HDF5 file."""
    # Mock h5py.File context manager
    mock_h5file = mocker.MagicMock()
    mock_h5_context = mocker.MagicMock()
    mock_h5_context.__enter__.return_value = mock_h5file
    mocker.patch("h5py.File", return_value=mock_h5_context)

    mock_read = mocker.patch(ASTROPY_TABLE_READ)
    mock_table = Table({"col1": [1, 2]})
    mock_read.return_value = mock_table

    mock_file_type = mocker.patch(READ_TABLE_FILE_TYPE)
    mock_file_type.return_value = "HDF5"

    result = read_tables(TEST_H5, ["table1", "table2"])

    assert len(result) == 2
    assert all(name in result for name in ["table1", "table2"])
    assert mock_read.call_count == 2
    mock_read.assert_has_calls(
        [mocker.call(TEST_H5, path="table1"), mocker.call(TEST_H5, path="table2")]
    )


def test_read_tables_unsupported_format(mocker):
    mock_file_type = mocker.patch(READ_TABLE_FILE_TYPE)
    mock_file_type.return_value = "CSV"

    with pytest.raises(ValueError, match="Unsupported file format"):
        read_tables(TEST_CSV, ["table1"])


def test_read_tables_explicit_file_type(mocker):
    mock_read = mocker.patch(ASTROPY_TABLE_READ)
    mock_table = Table({"col1": [1, 2]})
    mock_read.return_value = mock_table

    mock_file_type = mocker.patch(READ_TABLE_FILE_TYPE)

    result = read_tables(TEST_FITS, ["table1"], file_type="FITS")

    assert len(result) == 1
    mock_file_type.assert_not_called()
    mock_read.assert_called_once_with(TEST_FITS, hdu="table1")


def test_write_tables_fits(tmp_path, mock_table, mock_fits_objects):
    """Test writing tables in FITS format."""
    output_file = tmp_path / TEST_FITS

    write_tables([mock_table], output_file, file_type="FITS")

    mock_fits_objects["hdul"].assert_called_once()
    mock_fits_objects["hdul"].return_value.writeto.assert_called_once_with(
        output_file, checksum=False
    )


def test_write_tables_fits_overwrite_false(tmp_path, mock_table, mock_fits_objects, mocker):
    """Test writing tables in FITS format when overwrite is False and file exists."""
    output_file = tmp_path / TEST_FITS

    mocker.patch("pathlib.Path.exists", return_value=True)

    with pytest.raises(FileExistsError, match="^Output file "):
        write_tables([mock_table], output_file, overwrite_existing=False, file_type="FITS")


def test_write_tables_hdf5(tmp_path, mock_table, mock_h5py_file):
    """Test writing tables in HDF5 format."""
    output_file = tmp_path / TEST_H5
    write_tables([mock_table], output_file, file_type="HDF5")

    # Verify h5py operations were called correctly
    mock_h5py_file.create_dataset.assert_called_once()
    call_args = mock_h5py_file.create_dataset.call_args[1]
    assert call_args["compression"] == "gzip"
    assert call_args["compression_opts"] == 4


def test_write_tables_dict_input(tmp_path, mock_table, mock_h5py_file):
    """Test writing dictionary of tables."""
    tables_dict = {"table1": mock_table}
    output_file = tmp_path / TEST_H5
    write_tables(tables_dict, output_file, file_type="HDF5")

    # Verify h5py operations were called
    mock_h5py_file.create_dataset.assert_called_once()


def test_write_tables_existing_file(tmp_path, mocker):
    """Test writing tables when output file exists."""
    mock_table = Table({"col1": [1, 2]})
    mock_table.meta["EXTNAME"] = TEST_TABLE_NAME

    output_file = tmp_path / TEST_FITS
    output_file.touch()  # Create the file

    mocker.patch("astropy.io.fits.PrimaryHDU")
    mocker.patch("astropy.io.fits.BinTableHDU")
    mock_hdul = mocker.patch("astropy.io.fits.HDUList")

    write_tables([mock_table], output_file, file_type="FITS")

    assert not output_file.exists()  # File should be deleted before writing
    mock_hdul.assert_called_once()


def test_write_tables_no_file_type(tmp_path, mock_table, mock_read_type, mock_fits_objects):
    """Test writing tables without explicit file type."""
    mock_read_type.return_value = "FITS"
    output_file = tmp_path / TEST_FITS

    write_tables([mock_table], output_file)

    mock_read_type.assert_called_once_with([output_file])
    mock_fits_objects["hdul"].assert_called_once()


def test_merge_tables_success(mock_read_type, mock_logger, mocker):
    """Test successful table merging."""
    mock_merge = mocker.patch(f"{TABLE_HANDLER_PATH}._merge")
    mock_write = mocker.patch(f"{TABLE_HANDLER_PATH}.write_tables")

    input_files = [FILE1_FITS, FILE2_FITS]
    table_names = ["table1", "table2"]
    output_file = OUTPUT_FITS

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
    mock_read_type = mocker.patch(READ_TABLE_FILE_TYPE)
    mock_read_type.side_effect = ValueError("Test error")

    input_files = ["file1.txt"]
    table_names = ["table1"]
    output_file = OUTPUT_FITS

    with pytest.raises(ValueError, match="Test error"):
        merge_tables(input_files, table_names, output_file)


def test_copy_metadata_to_hdf5(mocker):
    """Test copying metadata between HDF5 files."""
    mock_h5py = mocker.patch(H5PY_FILE)
    mock_src = mocker.MagicMock()
    mock_dst = mocker.MagicMock()

    # Configure File context manager to return our mocks
    mock_h5py.return_value.__enter__.side_effect = [mock_src, mock_dst]

    # Set up source file mock with metadata
    table_name = TEST_TABLE_NAME
    meta_name = f"{table_name}.__table_column_meta__"
    mock_src.__contains__.return_value = True
    mock_dst.__contains__.return_value = False

    # Test copying metadata
    copy_metadata_to_hdf5(SOURCE_H5, DEST_H5, table_name)

    # Verify file opening modes
    mock_h5py.assert_any_call(SOURCE_H5, "r")
    mock_h5py.assert_any_call(DEST_H5, "a")

    # Verify metadata was copied
    mock_src.copy.assert_called_once_with(meta_name, mock_dst, name=meta_name)


def test_copy_metadata_to_hdf5_overwrite(mock_h5py_file, mocker):
    """Test copying metadata when it already exists in destination."""
    mock_src = mocker.MagicMock()
    mock_dst = mocker.MagicMock()

    mock_context = mocker.MagicMock()
    mock_context.__enter__.side_effect = [mock_src, mock_dst]
    mocker.patch(H5PY_FILE, return_value=mock_context)

    table_name = TEST_TABLE_NAME
    meta_name = f"{table_name}.__table_column_meta__"

    # Configure mocks to indicate metadata exists in both files
    mock_src.__contains__.return_value = True
    mock_dst.__contains__.return_value = True

    copy_metadata_to_hdf5(SOURCE_H5, DEST_H5, table_name)

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
    mocker.patch(H5PY_FILE, return_value=mock_context)

    table_name = TEST_TABLE_NAME

    # Configure mock to indicate no metadata in source
    mock_src.__contains__.return_value = False

    copy_metadata_to_hdf5(SOURCE_H5, DEST_H5, table_name)

    # Verify no copy operation was performed
    mock_src.copy.assert_not_called()
    mock_dst.__delitem__.assert_not_called()


def test_write_table_in_hdf5_new_table(mocker, mock_h5py_file, mock_table):
    """Test writing a new table to HDF5 file."""
    mock_h5py_file.__contains__.return_value = False
    mock_dataset = mocker.MagicMock()
    mock_h5py_file.create_dataset.return_value = mock_dataset

    write_table_in_hdf5(mock_table, TEST_H5, TEST_TABLE_NAME)

    # Verify dataset creation
    mock_h5py_file.create_dataset.assert_called_once()
    call_args = mock_h5py_file.create_dataset.call_args[1]
    assert call_args["compression"] == "gzip"
    assert call_args["compression_opts"] == 4
    assert call_args["chunks"] is True

    # Verify metadata was written
    assert mock_dataset.attrs.__setitem__.call_count == len(mock_table.meta)
    mock_dataset.attrs.__setitem__.assert_any_call("EXTNAME", TEST_TABLE_NAME)


def test_write_table_in_hdf5_append(mock_h5py_file, mock_table):
    """Test appending to existing table in HDF5 file."""
    mock_h5py_file.__contains__.return_value = True
    mock_dataset = mock_h5py_file.__getitem__.return_value
    mock_dataset.shape = (2, 2)  # Initial shape

    write_table_in_hdf5(mock_table, TEST_H5, TEST_TABLE_NAME)

    # Verify dataset was resized and data appended
    mock_dataset.resize.assert_called_once()
    mock_dataset.__setitem__.assert_called_once()


def test_write_table_in_hdf5_empty_table(mock_h5py_file):
    """Test writing empty table to HDF5 file."""
    empty_table = Table()
    empty_table.meta["EXTNAME"] = "empty_table"

    write_table_in_hdf5(empty_table, TEST_H5, "empty_table")

    # Verify dataset creation was called with empty data
    mock_h5py_file.create_dataset.assert_called_once()
    call_args = mock_h5py_file.create_dataset.call_args[1]
    assert np.array_equal(call_args["data"], np.array([]))


def test_read_table_list_hdf5(mocker):
    """Test read_table_list with HDF5 file."""
    mock_read_type = mocker.patch(READ_TABLE_FILE_TYPE, return_value="HDF5")
    mock_read_hdf5 = mocker.patch(
        "simtools.io.table_handler._read_table_list_hdf5",
        return_value={"table1": ["table1"], "table2": ["table2"]},
    )

    result = read_table_list(TEST_H5, ["table1", "table2"])

    mock_read_type.assert_called_once_with(TEST_H5)
    mock_read_hdf5.assert_called_once_with(TEST_H5, ["table1", "table2"], False)
    assert result == {"table1": ["table1"], "table2": ["table2"]}


def test_read_table_list_fits(mocker):
    """Test read_table_list with FITS file."""
    mock_read_type = mocker.patch(READ_TABLE_FILE_TYPE, return_value="FITS")
    mock_read_fits = mocker.patch(
        "simtools.io.table_handler._read_table_list_fits",
        return_value={"table1": ["table1"], "table2": ["table2"]},
    )

    result = read_table_list(TEST_FITS, ["table1", "table2"], True)

    mock_read_type.assert_called_once_with(TEST_FITS)
    mock_read_fits.assert_called_once_with(TEST_FITS, ["table1", "table2"], True)
    assert result == {"table1": ["table1"], "table2": ["table2"]}


def test_read_table_list_unsupported_format(mocker):
    """Test read_table_list with unsupported file format."""
    mock_read_type = mocker.patch(READ_TABLE_FILE_TYPE, return_value="CSV")

    result = read_table_list(TEST_CSV, ["table1"])

    mock_read_type.assert_called_once_with(TEST_CSV)
    assert result is None


def test_read_table_list_hdf5_basic(mocker, mock_h5py_file):
    """Test reading basic HDF5 table list without indexed tables."""
    # Mock datasets
    dataset1 = mocker.MagicMock(spec=h5py.Dataset)
    dataset2 = mocker.MagicMock(spec=h5py.Dataset)
    mock_h5py_file.visititems.side_effect = lambda x: [
        x("table1", dataset1),
        x("table2", dataset2),
    ]

    result = _read_table_list_hdf5(TEST_H5, ["table1", "table2"], False)

    assert result == {"table1": ["table1"], "table2": ["table2"]}


def test_read_table_list_hdf5_with_indexed(mocker, mock_h5py_file):
    """Test reading HDF5 table list with indexed tables."""
    # Mock datasets
    datasets = {
        "table1": mocker.MagicMock(spec=h5py.Dataset),
        "table1_0": mocker.MagicMock(spec=h5py.Dataset),
        "table1_1": mocker.MagicMock(spec=h5py.Dataset),
        "table2": mocker.MagicMock(spec=h5py.Dataset),
        "table2_0": mocker.MagicMock(spec=h5py.Dataset),
    }

    def mock_visititems(visitor):
        for name, dataset in datasets.items():
            visitor(name, dataset)

    mock_h5py_file.visititems.side_effect = mock_visititems

    result = _read_table_list_hdf5(TEST_H5, ["table1", "table2"], True)

    assert result == {
        "table1": ["table1", "table1_0", "table1_1"],
        "table2": ["table2", "table2_0"],
    }


def test_read_table_list_hdf5_ignore_non_datasets(mocker, mock_h5py_file):
    """Test that non-dataset objects are ignored."""
    # Mock a group (not a dataset)
    group = mocker.MagicMock(spec=h5py.Group)
    dataset = mocker.MagicMock(spec=h5py.Dataset)

    def mock_visititems(visitor):
        visitor("table1", group)  # Should be ignored
        visitor("table1", dataset)  # Should be included

    mock_h5py_file.visititems.side_effect = mock_visititems

    result = _read_table_list_hdf5(TEST_H5, ["table1"], False)

    assert result == {"table1": ["table1"]}


def test_read_table_list_hdf5_ignore_invalid_suffix(mocker, mock_h5py_file):
    """Test that indexed tables with invalid suffixes are ignored."""
    dataset1 = mocker.MagicMock(spec=h5py.Dataset)
    dataset2 = mocker.MagicMock(spec=h5py.Dataset)

    def mock_visititems(visitor):
        visitor("table1", dataset1)
        visitor("table1_abc", dataset2)  # Invalid suffix

    mock_h5py_file.visititems.side_effect = mock_visititems

    result = _read_table_list_hdf5(TEST_H5, ["table1"], True)

    assert result == {"table1": ["table1"]}


def test_read_table_list_hdf5_empty_file(mock_h5py_file):
    """Test reading from an empty HDF5 file."""
    mock_h5py_file.visititems.side_effect = lambda x: None

    result = _read_table_list_hdf5(TEST_H5, ["table1", "table2"], False)

    assert result == {"table1": [], "table2": []}


def test_read_table_list_fits_basic(mocker):
    """Test reading basic FITS table list without indexed tables."""
    mock_primary = mocker.MagicMock(spec=fits.PrimaryHDU)
    mock_primary.name = "PRIMARY"

    mock_table1 = mocker.MagicMock(spec=fits.BinTableHDU)
    mock_table1.name = "table1"
    mock_table1.is_image = False

    mock_table2 = mocker.MagicMock(spec=fits.BinTableHDU)
    mock_table2.name = "table2"
    mock_table2.is_image = False

    mock_table3 = mocker.MagicMock(spec=fits.BinTableHDU)
    mock_table3.name = "table3"
    mock_table3.is_image = False

    mock_table4 = mocker.MagicMock(spec=fits.TableHDU)
    mock_table4.name = "table4"
    mock_table4.is_image = False

    mock_hdul = [mock_primary, mock_table1, mock_table2, mock_table3, mock_table4]

    # Create a context manager mock that returns our HDU list
    mock_fits_open = mocker.MagicMock()
    mock_fits_open.__enter__ = mocker.Mock(return_value=mock_hdul)
    mock_fits_open.__exit__ = mocker.Mock(return_value=None)

    mocker.patch("astropy.io.fits.open", return_value=mock_fits_open)

    result = _read_table_list_fits(TEST_FITS, ["table1", "table2"], False)

    assert result == {"table1": ["table1"], "table2": ["table2"]}


def test_read_table_list_fits_with_indexed(mocker):
    """Test reading FITS table list with indexed tables."""
    mock_primary = mocker.MagicMock(spec=fits.PrimaryHDU)
    mock_primary.name = "PRIMARY"

    mock_table1 = mocker.MagicMock(spec=fits.BinTableHDU)
    mock_table1.name = "table1"
    mock_table1.is_image = False

    mock_table1_0 = mocker.MagicMock(spec=fits.BinTableHDU)
    mock_table1_0.name = "table1_0"
    mock_table1_0.is_image = False

    mock_table1_1 = mocker.MagicMock(spec=fits.BinTableHDU)
    mock_table1_1.name = "table1_1"
    mock_table1_1.is_image = False

    mock_table2 = mocker.MagicMock(spec=fits.BinTableHDU)
    mock_table2.name = "table2"
    mock_table2.is_image = False

    mock_table2_0 = mocker.MagicMock(spec=fits.BinTableHDU)
    mock_table2_0.name = "table2_0"
    mock_table2_0.is_image = False

    mock_hdul = [
        mock_primary,
        mock_table1,
        mock_table1_0,
        mock_table1_1,
        mock_table2,
        mock_table2_0,
    ]

    # Create a context manager mock that returns our HDU list
    mock_fits_open = mocker.MagicMock()
    mock_fits_open.__enter__ = mocker.Mock(return_value=mock_hdul)
    mock_fits_open.__exit__ = mocker.Mock(return_value=None)

    mocker.patch("astropy.io.fits.open", return_value=mock_fits_open)

    result = _read_table_list_fits(TEST_FITS, ["table1", "table2"], True)

    assert result == {
        "table1": ["table1", "table1_0", "table1_1"],
        "table2": ["table2", "table2_0"],
    }


def test_write_table_in_hdf5_unicode_conversion(mock_h5py_file):
    """Test writing table with Unicode columns to HDF5."""
    data = {"unicode_col": ["abc", "def", "ghi"]}
    table = Table(data)
    table.meta["EXTNAME"] = TEST_TABLE_NAME

    write_table_in_hdf5(table, TEST_H5, TEST_TABLE_NAME)

    mock_h5py_file.create_dataset.assert_called_once()
    call_args = mock_h5py_file.create_dataset.call_args[1]

    data_array = call_args["data"]
    assert isinstance(data_array, np.ndarray)
    for original in data["unicode_col"]:
        assert original.encode("ascii") in data_array.tobytes()


def test_write_table_in_hdf5_unicode_append(mock_h5py_file):
    """Test appending Unicode data to existing HDF5 dataset."""
    mock_h5py_file.__contains__.return_value = True
    mock_dataset = mock_h5py_file.__getitem__.return_value
    mock_dataset.shape = (2,)

    data = {"unicode_col": ["abc", "def", "ghi"]}
    table = Table(data)
    table.meta["EXTNAME"] = TEST_TABLE_NAME

    write_table_in_hdf5(table, TEST_H5, TEST_TABLE_NAME)

    mock_dataset.resize.assert_called_once()

    set_item_call = mock_dataset.__setitem__.call_args[0]
    appended_data = set_item_call[1]
    assert isinstance(appended_data, np.ndarray)
    for original in data["unicode_col"]:
        assert original.encode("ascii") in appended_data.tobytes()


def test_read_table_from_hdf5_basic(mocker):
    """Test basic reading from HDF5 without units."""
    # Mock Table.read
    mock_table = mocker.MagicMock(spec=Table)
    mock_table.colnames = ["col1"]
    mocker.patch(ASTROPY_TABLE_READ, return_value=mock_table)

    # Mock h5py.File context
    mock_file = mocker.MagicMock()
    mock_dataset = mocker.MagicMock()
    mock_dataset.attrs = {}
    mock_file.__getitem__.return_value = mock_dataset
    mock_context = mocker.MagicMock()
    mock_context.__enter__.return_value = mock_file
    mocker.patch(H5PY_FILE, return_value=mock_context)

    result = read_table_from_hdf5(TEST_H5, TEST_TABLE_NAME)

    # Verify basic calls
    mock_file.__getitem__.assert_called_once_with(TEST_TABLE_NAME)
    assert result == mock_table


def test_read_table_from_hdf5_with_units(mocker):
    """Test reading from HDF5 with unit attributes."""
    # Mock Table.read
    mock_table = mocker.MagicMock(spec=Table)
    mock_table.colnames = ["col1"]
    mocker.patch(ASTROPY_TABLE_READ, return_value=mock_table)

    # Mock h5py.File context
    mock_file = mocker.MagicMock()
    mock_dataset = mocker.MagicMock()
    mock_dataset.attrs = {"col1_unit": "m"}
    mock_file.__getitem__.return_value = mock_dataset
    mock_context = mocker.MagicMock()
    mock_context.__enter__.return_value = mock_file
    mocker.patch(H5PY_FILE, return_value=mock_context)

    result = read_table_from_hdf5(TEST_H5, TEST_TABLE_NAME)

    # Verify unit assignment
    mock_table.__getitem__.assert_called_once_with("col1")
    assert result == mock_table


def test_write_table_in_hdf5_with_units(mocker, mock_h5py_file):
    """Test writing table with units to HDF5."""
    table = Table({"col1": [1, 2] * u.m, "col2": [3, 4] * u.s})
    table.meta["EXTNAME"] = "table_with_units"

    mock_dataset = mocker.MagicMock()
    mock_h5py_file.create_dataset.return_value = mock_dataset
    mock_h5py_file.__contains__.return_value = False

    write_table_in_hdf5(table, TEST_H5, "table_with_units")

    # Verify unit attributes were set
    mock_dataset.attrs.__setitem__.assert_any_call("col1_unit", "m")
    mock_dataset.attrs.__setitem__.assert_any_call("col2_unit", "s")

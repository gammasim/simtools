"""IO operations on astropy tables."""

import logging
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack

from simtools.utils import general

_logger = logging.getLogger(__name__)


def _decode_hdf5_string_column(column):
    """Decode HDF5 byte-string columns to unicode."""
    if column.dtype.kind == "S":
        return column.astype(str)

    if column.dtype.kind == "O" and column.size:
        sample = column.flat[0]
        if isinstance(sample, (bytes, np.bytes_)):
            decoded = [
                item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else item
                for item in column
            ]
            return np.asarray(decoded, dtype=str)

    return column


def read_table_list(input_file, table_names, include_indexed_tables=False):
    """
    Read available tables found in the input file.

    If table_counter is True, search for tables with the same name
    but with different suffixes (e.g., "_0", "_1", etc.).

    """
    file_type = read_table_file_type(input_file)
    if file_type == "HDF5":
        return _read_table_list_hdf5(input_file, table_names, include_indexed_tables)
    if file_type == "FITS":
        return _read_table_list_fits(input_file, table_names, include_indexed_tables)
    return None


def _read_table_list_hdf5(input_file, table_names, include_indexed_tables):
    """Read available tables from HDF5 file."""
    datasets = {name: [] for name in table_names}

    def is_indexed_variant(name, base):
        if not name.startswith(f"{base}_"):
            return False
        suffix = name[len(base) + 1 :]
        return suffix.isdigit()

    def visitor(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return

        for base in datasets:
            if name == base or (include_indexed_tables and is_indexed_variant(name, base)):
                datasets[base].append(name)

    with h5py.File(input_file, "r") as f:
        f.visititems(visitor)

    return datasets


def _read_table_list_fits(input_file, table_names, include_indexed_tables):
    """Read available tables from FITS file."""
    datasets = {name: [] for name in table_names}

    with fits.open(input_file) as hdul:
        for hdu in hdul[1:]:
            if not isinstance(hdu, fits.BinTableHDU):
                continue
            name = hdu.name
            if name in table_names:
                datasets[name].append(name)
                continue
            if not include_indexed_tables or "_" not in name:
                continue
            base, _, suffix = name.rpartition("_")
            if base in table_names and suffix.isdigit():
                datasets[base].append(name)

    return datasets


def merge_tables(input_files, input_table_names, output_file):
    """
    Merge multiple astropy tables from different files into a single file.

    Handles multiple tables per file and supports both HDF5 and FITS formats.
    Updates 'file_id' column if present to maintain file origin tracking.

    Parameters
    ----------
    input_files : list of str
        List of input file paths to be merged.
    input_table_names : list of str
        List of table names to be merged from each input file.
    output_file : str
        Path to the output file where the merged data will be saved.

    Returns
    -------
    None
    """
    _logger.info(f"Merging {len(input_files)} files into {output_file}")

    file_type = read_table_file_type(input_files)
    merged_tables = _merge(input_files, input_table_names, file_type, output_file)
    if file_type != "HDF5":
        write_tables(merged_tables, output_file, file_type)


def read_table_file_type(input_files):
    """
    Determine the file type of the input files.

    All input files must be of the same type (either all HDF5 or all FITS).

    Parameters
    ----------
    input_files : list of str
        List of input file paths.

    Returns
    -------
    str
        File type ('HDF5' or 'FITS').
    """
    if not input_files:
        raise ValueError("No input files provided.")
    input_files = [input_files] if isinstance(input_files, str | Path) else input_files

    def get_type(f):
        if f.lower().endswith((".hdf5", ".h5")):
            return "HDF5"
        if f.lower().endswith((".fits", ".fits.gz")):
            return "FITS"
        raise ValueError(f"Unsupported file type: {f}")

    file_types = {get_type(str(f)) for f in input_files}
    if len(file_types) != 1:
        raise ValueError("All input files must be of the same type (either all HDF5 or all FITS)")
    return file_types.pop()


def _merge(input_files, table_names, file_type, output_file, add_file_id_to_table_name=True):
    """
    Merge tables from multiple input files into single tables.

    Parameters
    ----------
    input_files : list of str
        List of input file paths to be merged.
    table_names : list of str
        List of table names to be merged from each input file.
    file_type : str
        Type of the input files ('HDF5' or 'FITS').
    add_file_id_to_table_name : bool, optional
        If True, appends the file index to the table name.

    Returns
    -------
    dict
        Dictionary with table names as keys and merged astropy tables as values.
    """
    merged = {name: [] for name in table_names}
    is_hdf5 = file_type == "HDF5"

    def update_file_id(table, idx):
        if "file_id" in table.colnames:
            table["file_id"] = idx

    def process_table(table, key, idx):
        table_name = f"{key}_{idx}" if add_file_id_to_table_name else key
        update_file_id(table, idx)
        if is_hdf5:
            write_table_in_hdf5(table, output_file, table_name)
            if idx == 0:
                copy_metadata_to_hdf5(input_files[0], output_file, table_name)
        else:
            merged[key].append(table)

    for idx, file in enumerate(input_files):
        tables = read_tables(file, table_names, file_type)
        for key, table in tables.items():
            process_table(table, key, idx)

    if file_type != "HDF5":
        merged = {k: vstack(v, metadata_conflicts="silent") for k, v in merged.items()}

    return merged


def read_tables(file, table_names, file_type=None, table_columns=None):
    """
    Read tables from a file.

    Parameters
    ----------
    file : str
        Path to the input file.
    table_names : list of str
        List of table names to read.
    file_type : str
        Type of the input file ('HDF5' or 'FITS').
    table_columns : dict, optional
        Mapping of table name to list of columns to read. Used for HDF5 only.

    Returns
    -------
    dict
        Dictionary with table names as keys and astropy tables as values.
    """
    file_type = file_type or read_table_file_type([file])
    if file_type == "HDF5":
        return {
            name: read_table_from_hdf5(
                file,
                name,
                columns=(table_columns or {}).get(name),
            )
            for name in table_names
        }
    if file_type == "FITS":
        return {name: Table.read(file, hdu=name) for name in table_names}
    raise ValueError(f"Unsupported file format: {file_type}. Supported formats are HDF5 and FITS.")


def read_table_from_hdf5(file, table_name, columns=None):
    """
    Read a single astropy table from an HDF5 file.

    Parameters
    ----------
    file : str or Path
        Path to the input HDF5 file.
    table_name : str
        Name of the table to read.
    columns : list[str], optional
        List of columns to read from a compound HDF5 dataset. If None, all columns are read.

    Returns
    -------
    astropy.table.Table
        The requested astropy table.
    """
    table = None
    if columns is None:
        table = Table.read(file, path=table_name)

    with h5py.File(file, "r") as f:
        dset = f[table_name]

        if columns is not None:
            if dset.dtype.names is None:
                raise ValueError(
                    f"Table '{table_name}' does not use a compound dtype and "
                    "cannot be column-filtered."
                )

            missing_columns = [col for col in columns if col not in dset.dtype.names]
            if missing_columns:
                raise KeyError(f"Columns {missing_columns} not found in table '{table_name}'.")

            if hasattr(dset, "fields"):
                data = dset.fields(columns)[:]
            else:
                data = dset[:][columns]

            table = Table({col: _decode_hdf5_string_column(data[col]) for col in columns})
            table.meta["EXTNAME"] = table_name

        for col in table.colnames:
            unit_key = f"{col}_unit"
            if unit_key in dset.attrs:
                table[col].unit = u.Unit(dset.attrs[unit_key])
    return table


def write_tables(tables, output_file, overwrite_existing=True, file_type=None):
    """
    Write tables to file (overwriting if exists).

    HDF5 files are written to a sibling ``.incomplete-<uuid>`` file, validated,
    and atomically moved to the requested output path. If writing fails, the
    incomplete file is retained for diagnosis and any existing output remains
    unchanged.

    Parameters
    ----------
    tables : list or dict
        List or Dictionary with astropy tables as values.
    output_file : str or Path
        Path to the output file.
    overwrite_existing : bool
        If True, overwrite the output file if it exists.
    file_type : str
        Type of the output file ('HDF5' or 'FITS').

    Returns
    -------
    None
    """
    output_file = Path(output_file)
    file_type = file_type or read_table_file_type([output_file])
    if output_file.exists() and not overwrite_existing:
        raise FileExistsError(f"Output file {output_file} already exists.")
    if output_file.exists() and file_type != "HDF5":
        output_file.unlink()
    hdus = [fits.PrimaryHDU()]
    if isinstance(tables, dict):
        tables = list(tables.values())
    if file_type == "HDF5":
        _write_tables_atomically_to_hdf5(tables, output_file)
        return
    for table in tables:
        _table_name = table.meta.get("EXTNAME")
        if file_type == "FITS":
            _logger.info(f"Writing table {_table_name} of length {len(table)} to {output_file}")
            hdu = fits.table_to_hdu(table)
            hdu.name = _table_name
            hdus.append(hdu)

    if file_type == "FITS":
        fits.HDUList(hdus).writeto(output_file, checksum=False)


def _write_tables_atomically_to_hdf5(tables, output_file):
    """Write and validate an HDF5 file before publishing it atomically."""
    write_table_chunks([tables], output_file)


def write_table_chunks(table_chunks, output_file, overwrite_existing=True):
    """Write table chunks to an atomic HDF5 output with bounded memory use."""
    output_file = Path(output_file)
    if output_file.exists() and not overwrite_existing:
        raise FileExistsError(f"Output file {output_file} already exists.")
    incomplete_file = output_file.with_name(f"{output_file.name}.incomplete-{general.get_uuid()}")
    expected_tables = {}

    try:
        with h5py.File(incomplete_file, "w") as hdf5_file:
            hdf5_file.attrs["simtools_write_status"] = "incomplete"
            for tables in table_chunks:
                chunk_table_names = set()
                for table in tables:
                    table_name = table.meta.get("EXTNAME")
                    if not table_name:
                        raise ValueError("Cannot write table without an 'EXTNAME' metadata value.")
                    if table_name in chunk_table_names:
                        raise ValueError(f"Duplicate output table name '{table_name}'.")
                    chunk_table_names.add(table_name)
                    expected_tables[table_name] = expected_tables.get(table_name, 0) + len(table)
                    _write_table_to_hdf5_file(table, hdf5_file, table_name)

        _validate_written_hdf5(incomplete_file, expected_tables)
        with h5py.File(incomplete_file, "r+") as hdf5_file:
            hdf5_file.attrs["simtools_write_status"] = "complete"
            hdf5_file.flush()
        incomplete_file.replace(output_file)
        _logger.info(f"Published complete HDF5 output file {output_file}")
    except Exception:
        _logger.exception(
            f"Failed to publish HDF5 output file '{output_file}'. "
            f"Incomplete output, if created, is stored at '{incomplete_file}'."
        )
        raise


def _validate_written_hdf5(output_file, expected_tables):
    """Verify that all requested tables were persisted with the expected row counts."""
    if not expected_tables:
        raise ValueError("Cannot publish an HDF5 file without tables.")

    with h5py.File(output_file, "r") as hdf5_file:
        for table_name, expected_rows in expected_tables.items():
            if table_name not in hdf5_file:
                raise ValueError(f"Output table '{table_name}' was not written.")
            dataset = hdf5_file[table_name]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(f"Output object '{table_name}' is not an HDF5 dataset.")
            if dataset.dtype.names is None:
                raise ValueError(f"Output table '{table_name}' has no compound dtype.")
            if len(dataset) != expected_rows:
                raise ValueError(
                    f"Output table '{table_name}' has {len(dataset)} row(s), "
                    f"expected {expected_rows}."
                )


def _prepare_string_columns_for_hdf5(table):
    """Convert supported string columns to the byte strings required by HDF5."""
    string_columns = []
    for column_name in table.colnames:
        column = table[column_name]
        if column.dtype.kind == "U":  # hdf5 does not support unicode
            string_columns.append(column_name)
        elif column.dtype.kind == "O":
            values = np.asarray(column)
            if not all(
                isinstance(value, (str, bytes, np.str_, np.bytes_)) for value in values.flat
            ):
                raise TypeError(
                    f"Object-dtype column '{column_name}' contains non-string or missing values; "
                    "refusing to serialize it as strings."
                )
            string_columns.append(column_name)

    if not string_columns:
        return table

    table = table.copy(copy_data=False)
    for column_name in string_columns:
        table[column_name] = table[column_name].astype("S")
    return table


def write_table_in_hdf5(table, output_file, table_name):
    """
    Write or append a single astropy table to an HDF5 file.

    Parameters
    ----------
    table : astropy.table.Table
        The astropy table to write.
    output_file : str or Path
        Path to the output HDF5 file.
    table_name : str
        Name of the table in the HDF5 file.

    Returns
    -------
    None
    """
    with h5py.File(output_file, "a") as f:
        _write_table_to_hdf5_file(table, f, table_name)


def _write_table_to_hdf5_file(table, hdf5_file, table_name):
    """Write or append one table using an already open HDF5 file."""
    table = _prepare_string_columns_for_hdf5(table)
    data = np.array(table)
    if table_name not in hdf5_file:
        dset = _create_hdf5_dataset(hdf5_file, table_name, data)
        for key, val in table.meta.items():
            dset.attrs[key] = val
        for col in table.colnames:
            unit = getattr(table[col], "unit", None)
            if unit is not None:
                dset.attrs[f"{col}_unit"] = str(unit)
        return
    if len(data) == 0:
        return

    dset = hdf5_file[table_name]
    promoted_dtype = np.promote_types(dset.dtype, data.dtype)
    if promoted_dtype != dset.dtype:
        dset = _replace_dataset_with_promoted_dtype(hdf5_file, table_name, promoted_dtype)
    old_length = len(dset)
    dset.resize(old_length + len(data), axis=0)
    dset[old_length:] = data


def _create_hdf5_dataset(hdf5_file, table_name, data=None, dtype=None, shape=None):
    """Create a compressed, extensible HDF5 dataset."""
    dataset_shape = data.shape if data is not None else shape
    kwargs = {
        "maxshape": (None, *dataset_shape[1:]),
        "chunks": True,
        "compression": "gzip",
        "compression_opts": 4,
    }
    if data is not None:
        kwargs["data"] = data
    else:
        kwargs.update({"shape": shape, "dtype": dtype})
    return hdf5_file.create_dataset(table_name, **kwargs)


def _replace_dataset_with_promoted_dtype(hdf5_file, table_name, dtype):
    """Replace a dataset while widening fields such as fixed-length strings."""
    old_dataset = hdf5_file[table_name]
    temporary_name = f"{table_name}.__promoted__"
    new_dataset = _create_hdf5_dataset(
        hdf5_file,
        temporary_name,
        dtype=dtype,
        shape=old_dataset.shape,
    )
    for key, value in old_dataset.attrs.items():
        new_dataset.attrs[key] = value
    rows_per_chunk = old_dataset.chunks[0] if old_dataset.chunks else 100_000
    for start in range(0, len(old_dataset), rows_per_chunk):
        new_dataset[start : start + rows_per_chunk] = old_dataset[start : start + rows_per_chunk]
    del hdf5_file[table_name]
    hdf5_file.move(temporary_name, table_name)
    return hdf5_file[table_name]


def copy_metadata_to_hdf5(src_file, dst_file, table_name):
    """
    Copy metadata (table column meta) from one HDF5 file to another.

    For merging tables, this function ensures that the metadata is preserved.

    Parameters
    ----------
    src_file : str or Path
        Path to the source HDF5 file.
    dst_file : str or Path
        Path to the destination HDF5 file.
    table_name : str
        Name of the table whose metadata is to be copied.
    """
    with h5py.File(src_file, "r") as src, h5py.File(dst_file, "a") as dst:
        meta_name = f"{table_name}.__table_column_meta__"
        if meta_name in src:
            if meta_name in dst:
                del dst[meta_name]  # overwrite if exists
            src.copy(meta_name, dst, name=meta_name)

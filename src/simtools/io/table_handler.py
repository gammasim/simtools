"""IO operations on astropy tables."""

import logging
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack

_logger = logging.getLogger(__name__)


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


def read_tables(file, table_names, file_type=None):
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

    Returns
    -------
    dict
        Dictionary with table names as keys and astropy tables as values.
    """
    file_type = file_type or read_table_file_type([file])
    if file_type == "HDF5":
        return {name: read_table_from_hdf5(file, name) for name in table_names}
    if file_type == "FITS":
        return {name: Table.read(file, hdu=name) for name in table_names}
    raise ValueError(f"Unsupported file format: {file_type}. Supported formats are HDF5 and FITS.")


def read_table_from_hdf5(file, table_name):
    """
    Read a single astropy table from an HDF5 file.

    Parameters
    ----------
    file : str or Path
        Path to the input HDF5 file.
    table_name : str
        Name of the table to read.

    Returns
    -------
    astropy.table.Table
        The requested astropy table.
    """
    table = Table.read(file, path=table_name)
    with h5py.File(file, "r") as f:
        dset = f[table_name]
        for col in table.colnames:
            unit_key = f"{col}_unit"
            if unit_key in dset.attrs:
                table[col].unit = u.Unit(dset.attrs[unit_key])
    return table


def write_tables(tables, output_file, overwrite_existing=True, file_type=None):
    """
    Write tables to file (overwriting if exists).

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
    if output_file.exists():
        if overwrite_existing:
            output_file.unlink()
        else:
            raise FileExistsError(f"Output file {output_file} already exists.")
    hdus = [fits.PrimaryHDU()]
    if isinstance(tables, dict):
        tables = list(tables.values())
    for table in tables:
        _table_name = table.meta.get("EXTNAME")
        _logger.info(f"Writing table {_table_name} of length {len(table)} to {output_file}")
        if file_type == "HDF5":
            write_table_in_hdf5(table, output_file, _table_name)
        if file_type == "FITS":
            hdu = fits.table_to_hdu(table)
            hdu.name = _table_name
            hdus.append(hdu)

    if file_type == "FITS":
        fits.HDUList(hdus).writeto(output_file, checksum=False)


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
    for col in table.colnames:
        if table[col].dtype.kind == "U":  # hdf5 does not support unicode
            table[col] = table[col].astype("S")

    with h5py.File(output_file, "a") as f:
        data = np.array(table)
        if table_name not in f:
            maxshape = (None, *data.shape[1:])
            dset = f.create_dataset(
                table_name,
                data=data,
                maxshape=maxshape,
                chunks=True,
                compression="gzip",
                compression_opts=4,
            )
            for key, val in table.meta.items():
                dset.attrs[key] = val
            for col in table.colnames:
                unit = getattr(table[col], "unit", None)
                if unit is not None:
                    dset.attrs[f"{col}_unit"] = str(unit)
        else:
            dset = f[table_name]
            dset.resize(dset.shape[0] + data.shape[0], axis=0)
            dset[-data.shape[0] :] = data


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

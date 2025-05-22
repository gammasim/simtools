"""IO operations on astropy tables."""

import importlib.util
import logging
from pathlib import Path

from astropy.io import fits
from astropy.table import Table, vstack

_logger = logging.getLogger(__name__)


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
    merged_tables = _merge(input_files, input_table_names, file_type)
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

    def get_type(f):
        if f.lower().endswith((".hdf5", ".h5")):
            return "HDF5"
        if f.lower().endswith((".fits", ".fits.gz")):
            return "FITS"
        raise ValueError(f"Unsupported file type: {f}")

    file_types = {get_type(str(f)) for f in input_files}
    if len(file_types) != 1:
        raise ValueError("All input files must be of the same type (either all HDF5 or all FITS)")
    file_type = file_types.pop()
    if file_type == "HDF5" and importlib.util.find_spec("h5py") is None:
        raise ImportError("h5py is required to write HDF5 files with Astropy.")

    return file_type


def _merge(input_files, table_names, file_type):
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

    Returns
    -------
    dict
        Dictionary with table names as keys and merged astropy tables as values.
    """
    merged = {name: [] for name in table_names}

    for idx, file in enumerate(input_files):
        tables = read_tables(file, table_names, file_type)
        for key, table in tables.items():
            if "file_id" in table.colnames:  # update file file_id
                table["file_id"] = idx
            merged[key].append(table)

    for key in merged:
        merged[key] = vstack(merged[key], metadata_conflicts="silent")

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
        return {name: Table.read(file, path=name) for name in table_names}
    if file_type == "FITS":
        return {name: Table.read(file, hdu=name) for name in table_names}
    raise ValueError(f"Unsupported file format: {file_type}. Supported formats are HDF5 and FITS.")


def write_tables(tables, output_file, file_type=None):
    """
    Write tables to file (overwriting if exists).

    Parameters
    ----------
    tables : list or dict
        List or Dictionary with astropy tables as values.
    output_file : str or Path
        Path to the output file.
    file_type : str
        Type of the output file ('HDF5' or 'FITS').

    Returns
    -------
    None
    """
    output_file = Path(output_file)
    file_type = file_type or read_table_file_type([output_file])
    if output_file.exists():
        output_file.unlink()
    hdus = [fits.PrimaryHDU()]
    if isinstance(tables, dict):
        tables = list(tables.values())
    for table in tables:
        _table_name = table.meta.get("EXTNAME")
        _logger.info(f"Writing table {_table_name} of length {len(table)} to {output_file}.")
        if file_type == "HDF5":
            table.write(
                output_file,
                path=f"{_table_name}",
                append=True,
                format="hdf5",
                serialize_meta=True,
                compression=True,
            )
        if file_type == "FITS":
            hdu = fits.table_to_hdu(table)
            hdu.name = _table_name
            hdus.append(hdu)

    if file_type == "FITS":
        fits.HDUList(hdus).writeto(output_file, checksum=False)

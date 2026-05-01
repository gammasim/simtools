"""Utilities for exporting model parameter values / files from the database."""

from pathlib import Path

from simtools.data_model import row_table_utils
from simtools.simtel import simtel_table_reader

ECSV_SUFFIX = ".ecsv"


def _is_dict_table_value(parameter_info):
    """Return True if a parameter stores embedded row-oriented table data."""
    return parameter_info.get("type") == "dict" and row_table_utils.is_row_table_dict(
        parameter_info.get("value")
    )


def _get_parameter_info(
    db,
    parameter,
    site,
    array_element_name,
    parameter_version=None,
    model_version=None,
):
    """Fetch single-parameter metadata dict from DB."""
    parameters = db.get_model_parameter(
        parameter,
        site,
        array_element_name,
        parameter_version=parameter_version,
        model_version=model_version,
    )
    return parameters, parameters[parameter]


def _normalize_file_names(file_names=None, parameters=None):
    """Normalize file_names input or derive it from parameter metadata."""
    if file_names:
        return [file_names] if not isinstance(file_names, list) else file_names
    if parameters:
        return [
            info["value"]
            for info in parameters.values()
            if isinstance(info, dict) and info.get("file") and info.get("value") is not None
        ]
    return []


def write_file_from_db_to_disk(db, db_name, path, file):
    """
    Write one file object from GridFS to disk.

    Parameters
    ----------
    db : DatabaseHandler
        Database handler wrapper.
    db_name : str
        Database name.
    path : str or Path
        Output directory.
    file : gridfs.grid_file.GridOut
        File object returned by GridFS.
    """
    db.mongo_db_handler.write_file_from_db_to_disk(db_name, path, file)


def export_model_files(db, parameters=None, file_names=None, dest=None, db_name=None):
    """
    Export model files from DB to a destination directory.

    Parameters
    ----------
    db : DatabaseHandler
        Database handler wrapper.
    parameters : dict, optional
        Parameter metadata dictionary used to derive file names.
    file_names : str or list[str], optional
        File name or list of file names to export.
    dest : str or Path
        Output directory.
    db_name : str, optional
        Database name. Uses ``db.db_name`` when omitted.

    Returns
    -------
    dict
        Mapping of file name to GridFS id or ``"file exists"``.

    Raises
    ------
    ValueError
        If ``dest`` is not provided.
    """
    if dest is None:
        raise ValueError("Destination path is required to export model files.")

    db_name = db_name or db.db_name
    file_names = _normalize_file_names(file_names=file_names, parameters=parameters)
    destination = Path(dest)

    instance_ids = {}
    for file_name in file_names:
        if destination.joinpath(file_name).exists():
            instance_ids[file_name] = "file exists"
        else:
            file_path_instance = db.mongo_db_handler.get_file_from_db(db_name, file_name)
            db.write_file_from_db_to_disk(db_name, dest, file_path_instance)
            instance_ids[file_name] = file_path_instance._id  # pylint: disable=protected-access
    return instance_ids


def _export_dict_table_parameter(
    db,
    parameter,
    site,
    array_element_name,
    output_file,
    par_info,
    parameters,
    parameter_version=None,
    model_version=None,
):
    """
    Export dict-typed (embedded table) parameter to ECSV file.

    Returns the output file path.
    """
    if output_file is None:
        raise ValueError(
            "Use --output_file when exporting dict-typed parameters with "
            "--export_model_file or --export_model_file_as_table."
        )

    table = export_single_model_file(
        db=db,
        parameter=parameter,
        site=site,
        array_element_name=array_element_name,
        parameter_version=parameter_version,
        model_version=model_version,
        export_file_as_table=True,
        parameters=parameters,
        par_info=par_info,
    )
    table_file = db.io_handler.get_output_file(output_file).with_suffix(ECSV_SUFFIX)
    table.write(table_file, format="ascii.ecsv", overwrite=True)
    return [table_file]


def _export_file_backed_parameter(
    db,
    parameter,
    site,
    array_element_name,
    output_file,
    par_info,
    parameters,
    export_model_file_as_table,
    parameter_version=None,
    model_version=None,
):
    """
    Export file-backed parameter to disk.

    Exports the file and optionally also as an ECSV table.
    """
    table = export_single_model_file(
        db=db,
        parameter=parameter,
        site=site,
        array_element_name=array_element_name,
        parameter_version=parameter_version,
        model_version=model_version,
        export_file_as_table=export_model_file_as_table,
        parameters=parameters,
        par_info=par_info,
    )
    source_file = db.io_handler.get_output_file(par_info["value"])
    table_file = db.io_handler.get_output_file(output_file) if output_file else source_file
    if table_file != source_file:
        source_file.rename(table_file)
    output_files = [table_file]

    if table and table_file.suffix != ECSV_SUFFIX:
        table_output_file = table_file.with_suffix(ECSV_SUFFIX)
        table.write(table_output_file, format="ascii.ecsv", overwrite=True)
        output_files.append(table_output_file)

    return output_files


def export_single_model_file(
    db,
    parameter,
    site,
    array_element_name,
    model_version=None,
    parameter_version=None,
    export_file_as_table=False,
    parameters=None,
    par_info=None,
):
    """
    Export one parameter payload and optionally return it as a table.

    Parameters
    ----------
    db : DatabaseHandler
        Database handler wrapper.
    parameter : str
        Parameter name.
    site : str
        Site name.
    array_element_name : str
        Array element name.
    model_version : str, optional
        Model version.
    parameter_version : str, optional
        Parameter version.
    export_file_as_table : bool, optional
        If True, return an ``astropy.table.Table`` when possible.
    parameters : dict, optional
        Prefetched parameter dictionary.
    par_info : dict, optional
        Prefetched single-parameter entry.

    Returns
    -------
    astropy.table.Table or None
        Exported table when requested and available, otherwise None.
    """
    if parameters is None or par_info is None:
        parameters, par_info = _get_parameter_info(
            db=db,
            parameter=parameter,
            site=site,
            array_element_name=array_element_name,
            parameter_version=parameter_version,
            model_version=model_version,
        )

    if _is_dict_table_value(par_info):
        if export_file_as_table:
            return simtel_table_reader.row_data_to_astropy_table(par_info["value"])
        return None

    db.export_model_files(parameters=parameters, dest=db.io_handler.get_output_directory())
    if export_file_as_table:
        return simtel_table_reader.read_simtel_table(
            parameter,
            db.io_handler.get_output_directory().joinpath(par_info["value"]),
        )
    return None


def export_parameter_data(
    db,
    parameter,
    site,
    array_element_name,
    parameter_version=None,
    model_version=None,
    output_file=None,
    export_model_file=False,
    export_model_file_as_table=False,
):
    """
    Export parameter payload based on type and export flags.

    Parameters
    ----------
    db : DatabaseHandler
        DatabaseHandler instance used for DB access and file output.
    parameter : str
        Name of the parameter.
    site : str
        Site name.
    array_element_name : str
        Name of the array element model (e.g. LSTN-01).
    parameter_version : str, optional
        Version of the parameter.
    model_version : str, optional
        Version of the model.
    output_file : str, optional
        Output file name for dict-backed table exports and optional override
        for file-backed exports.
    export_model_file : bool, optional
        Export payload to files.
    export_model_file_as_table : bool, optional
        Also export file-backed payload as ECSV table.

    Returns
    -------
    list[Path]
        Output file paths.

    Raises
    ------
    ValueError
        If an incompatible combination of options is provided.
    """
    if export_model_file_as_table and not export_model_file:
        raise ValueError("Use --export_model_file together with --export_model_file_as_table.")

    if not (export_model_file or export_model_file_as_table):
        return []

    parameters, par_info = _get_parameter_info(
        db=db,
        parameter=parameter,
        site=site,
        array_element_name=array_element_name,
        parameter_version=parameter_version,
        model_version=model_version,
    )

    # Dispatch to appropriate export handler based on parameter type
    if _is_dict_table_value(par_info):
        return _export_dict_table_parameter(
            db=db,
            parameter=parameter,
            site=site,
            array_element_name=array_element_name,
            output_file=output_file,
            par_info=par_info,
            parameters=parameters,
            parameter_version=parameter_version,
            model_version=model_version,
        )

    return _export_file_backed_parameter(
        db=db,
        parameter=parameter,
        site=site,
        array_element_name=array_element_name,
        output_file=output_file,
        par_info=par_info,
        parameters=parameters,
        export_model_file_as_table=export_model_file_as_table,
        parameter_version=parameter_version,
        model_version=model_version,
    )

"""Utilities for exporting model parameter payloads."""

from pathlib import Path

from simtools.simtel import simtel_table_reader

ECSV_SUFFIX = ".ecsv"


def write_file_from_db_to_disk(db, db_name, path, file):
    """Extract a file from MongoDB and write it to disk."""
    db.mongo_db_handler.write_file_from_db_to_disk(db_name, path, file)


def export_model_files(db, parameters=None, file_names=None, dest=None, db_name=None):
    """Export model files from DB to the given directory."""
    db_name = db_name or db.db_name

    if file_names:
        file_names = [file_names] if not isinstance(file_names, list) else file_names
    elif parameters:
        file_names = [
            info["value"]
            for info in parameters.values()
            if info and info.get("file") and info["value"] is not None
        ]

    instance_ids = {}
    for file_name in file_names:
        if Path(dest).joinpath(file_name).exists():
            instance_ids[file_name] = "file exists"
        else:
            file_path_instance = db.mongo_db_handler.get_file_from_db(db_name, file_name)
            db.write_file_from_db_to_disk(db_name, dest, file_path_instance)
            instance_ids[file_name] = file_path_instance._id  # pylint: disable=protected-access
    return instance_ids


def export_single_model_file(
    db,
    parameter,
    site,
    array_element_name,
    model_version=None,
    parameter_version=None,
    export_file_as_table=False,
):
    """Export a single model file from DB identified by a parameter name."""
    parameters = db.get_model_parameter(
        parameter,
        site,
        array_element_name,
        parameter_version=parameter_version,
        model_version=model_version,
    )
    par_info = parameters[parameter]

    if par_info.get("type") == "dict" and isinstance(par_info.get("value"), dict):
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
    """Export model parameter payload based on parameter type and export flags.

    Parameters
    ----------
    db: DatabaseHandler
        DatabaseHandler instance used for DB access and file output.
    parameter: str
        Name of the parameter.
    site: str
        Site name.
    array_element_name: str
        Name of the array element model (e.g. LSTN-01).
    parameter_version: str
        Version of the parameter.
    model_version: str
        Version of the model.
    output_file: str
        Output file name for dict-backed table exports.
    export_model_file: bool
        Export payload to files.
    export_model_file_as_table: bool
        Also export file-backed payload as ECSV table.

    Returns
    -------
    list
        List of output file paths.

    Raises
    ------
    ValueError
        If an incompatible combination of options is provided.
    """
    if export_model_file_as_table and not export_model_file:
        raise ValueError("Use --export_model_file together with --export_model_file_as_table.")

    if not (export_model_file or export_model_file_as_table):
        return []

    pars = db.get_model_parameter(
        parameter=parameter,
        site=site,
        array_element_name=array_element_name,
        parameter_version=parameter_version,
        model_version=model_version,
    )
    par_info = pars[parameter]

    if par_info.get("type") == "dict" and isinstance(par_info.get("value"), dict):
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
        )
        table_file = db.io_handler.get_output_file(output_file).with_suffix(ECSV_SUFFIX)
        table.write(table_file, format="ascii.ecsv", overwrite=True)
        return [table_file]

    if output_file is not None:
        raise ValueError(
            "Do not use --output_file when exporting file-backed parameters with "
            "--export_model_file. The original database file name is used."
        )

    table = export_single_model_file(
        db=db,
        parameter=parameter,
        site=site,
        array_element_name=array_element_name,
        parameter_version=parameter_version,
        model_version=model_version,
        export_file_as_table=export_model_file_as_table,
    )
    param_value = par_info["value"]
    table_file = db.io_handler.get_output_file(param_value)
    output_files = [table_file]
    if table and table_file.suffix != ECSV_SUFFIX:
        table_output_file = table_file.with_suffix(ECSV_SUFFIX)
        table.write(table_output_file, format="ascii.ecsv", overwrite=True)
        output_files.append(table_output_file)

    return output_files

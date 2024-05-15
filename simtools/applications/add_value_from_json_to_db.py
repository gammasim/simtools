#!/usr/bin/python3

"""
Summary
-------
This application adds a new parameter / value to a collection in the DB using a json
file as input.

Command line arguments
----------------------
file_name (str, required)
Name of the file to upload including the full path.
db_collection (str, required)
The DB collection to which to add the file.
db (str)
The DB to insert the files to. \
        The choices are either the default CTA simulation DB or a sandbox for testing.

Example
-------

Upload a file to sites collection:

.. code-block:: console

simtools-add-value-from-json-to-db \
            --file_name new_value.json --db_collection sites


"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.db import db_handler


def main():
    config = configurator.Configurator(description="Add a new parameter to the DB.")
    group = config.parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file_name", help="file to be added", type=str)
    group.add_argument(
        "--input_path",
        help="A directory with json files to upload to the DB.",
        type=Path,
    )
    config.parser.add_argument(
        "--db_collection", help="DB collection to which to add new values.", required=True
    )
    config.parser.add_argument(
        "--db",
    )
    args_dict, db_config = config.initialize(db_config=True)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    db = db_handler.DatabaseHandler(mongo_db_config=db_config)

    files_to_insert = []
    if args_dict.get("file_name", None) is not None:
        files_to_insert.append(args_dict["file_name"])
    else:
        files_to_insert.extend(Path(args_dict["input_path"]).glob("*json"))

    if len(files_to_insert) < 1:
        raise ValueError("No files were provided to upload")
    plural = "s" if len(files_to_insert) > 1 else ""

    print(
        f"Should the following parameter{plural} be inserted to the "
        f"{args_dict['db_collection']} DB collection?:\n"
    )
    print(*files_to_insert, sep="\n")
    print()

    if gen.user_confirm():
        for file_to_insert_now in files_to_insert:
            par_dict = gen.collect_data_from_file_or_dict(
                file_name=file_to_insert_now, in_dict=None
            )
            logger.info(f"Adding the following parameter to the DB: {par_dict['parameter']}")
            db.add_new_parameter(
                db_name=db_config["db_simulation_model"],
                telescope=par_dict["instrument"],
                parameter=par_dict["parameter"],
                version=par_dict["version"],
                value=par_dict["value"],
                site=par_dict["site"],
                type=par_dict["type"],
                collection_name=args_dict["db_collection"],
                applicable=par_dict["applicable"],
                file=par_dict["file"],
                unit=par_dict.get("unit", None),
                file_prefix="./",
            )
            logger.info(
                f"Value for {par_dict['parameter']} added to "
                f"{args_dict['db_collection']} collection."
            )
    else:
        logger.info("Aborted, no change applied to the database")


if __name__ == "__main__":
    main()

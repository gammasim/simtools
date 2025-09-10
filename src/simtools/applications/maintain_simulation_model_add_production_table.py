r"""
Generate simulation model production tables for a new model version.

This script should be used to maintain the simulation model repository. It allows to create
new production tables by copying an existing base version and applies modifications defined in
a YAML file (see the example file below).

Two main use cases are covered by this script:

1. full_update: Create a complete new set of production tables (e.g. for new major or minor
   versions of the simulation models). This will copy all production tables from the source
   directory and apply the modifications to the tables that are listed in the modifications file.

2. patch_update: Create a set of new production tables including the changes defined in the
   modifications file. No unmodified tables are copied. For new production tables with patch
   modifications, the key-value pair `base_model_version: <base_model version>` is added.

Example
-------

The following example applies a patch update with changes defined in a YAML file.

.. code-block:: console

    simtools-maintain-simulation-model-add-production-table \\
        --simulation_models_path ../simulation-models-dev/simulation-models/ \\
        --base_model_version 6.0.0 \\
        --modifications tests/resources/production_tables_changes_for_threshold_study_6.2.0.yml \\
        --patch_update

"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.model import model_repository


def _parse(label, description):
    """
    Parse command line arguments.

    Returns
    -------
    dict
        Parsed command-line arguments.
    """
    config = configurator.Configurator(label=label, description=description)
    config.parser.add_argument(
        "--simulation_models_path",
        type=str,
        required=True,
        help="Path to the simulation models repository.",
    )
    config.parser.add_argument(
        "--base_model_version",
        type=str,
        required=True,
        help="Base model version (which is the source production table subdirectory to copy from).",
    )
    config.parser.add_argument(
        "--modifications",
        type=str,
        required=True,
        help="File containing the list of changes to apply.",
    )
    update_group = config.parser.add_mutually_exclusive_group(required=True)
    update_group.add_argument(
        "--full_update",
        action="store_true",
        default=False,
        help=(
            "Create a full new set of production tables by copying all tables from the "
            "base version and applying the modifications to the relevant tables."
        ),
    )
    update_group.add_argument(
        "--patch_update",
        action="store_true",
        default=False,
        help=(
            "Create a new set of production tables including only the changes defined in the "
            "modifications file. No unmodified tables are copied."
        ),
    )

    return config.initialize(db_config=False, output=False)


def main():  # noqa: D103
    label = Path(__file__).stem
    args_dict, _ = _parse(
        label=label, description=("Copy and update simulation model production tables.")
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    model_repository.copy_and_update_production_table(args_dict)


if __name__ == "__main__":
    main()

r"""
Generate new simulation model production tables by copying existing table and apply modifications.

This script should be used to maintain the simulation model repository. It allow to create a
new production table by copying an existing base version and apply modification defined in a YAML
file (see the example file listed below).

Example
-------
.. code-block:: console

    simtools-maintain-simulation-model-add-production-table \\
        --simulation_models_path ../simulation-models-dev/simulation-models/ \\
        --source_prod_table_dir 6.0.0 \\
        --modifications tests/resources/production_tables_changes_for_threshold_study_6.2.0.yml

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
        "--source_prod_table_dir",
        type=str,
        required=True,
        help="The source production table directory to copy from.",
    )
    config.parser.add_argument(
        "--modifications",
        type=str,
        required=True,
        help="File containing the list of changes to apply.",
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

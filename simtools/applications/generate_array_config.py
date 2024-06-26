#!/usr/bin/python3
"""
    Generate sim_telarray config files for a given array.

    All the input required has to be given as a yaml file by the command
    line argument array_config.

    The required entries in the array_config file are:

    'site': South or North

    'layout_name': name of an array layout.

    'model_version': model version.

    'default': telescope model names to be assigned to each telescope size by default.

    As optional data, specific telescope models can be assigned to individual telescopes.
    This is done by the entries with the name of the telescope (as used by the layout
    definition, ex. LSTN-01, MSTN-05, SSTS-10).

    Each telescope model can be set in two ways.

    a) A single str with the name of telescope model
    Ex. 'MSTN-05': 'NectarCam-D'

    b) A dict containing a 'name' key with the name of the telescope model and further keys
    with model parameters to be changed from the original model.
    Ex.:

    .. code-block:: python

        'MSTN-05': {
                'name': 'design',
                'fadc_pulse_shape': 'Pulse_template_nectarCam_17042020-noshift.dat',
                'discriminator_pulse_shape': 'Pulse_template_nectarCam_17042020-noshift.dat'
            }

    This is an example of the content of an array_config file.

    .. code-block:: python

        site: North,
        layout_name: Prod5
        model_version: Prod5
        default:
            LST: 'D234'  # Design model for the LSTs
            MST: FlashCam-D  # Design model for the MST-FlashCam
        LST-01: '1'  # Model of LST-01 in the LaPalma site.
        MST-05:
            name: NectarCam-D
            # Parameters to be changed
            fadc_pulse_shape: Pulse_template_nectarCam_17042020-noshift.dat
            discriminator_pulse_shape: Pulse_template_nectarCam_17042020-noshift.dat

    Command line arguments
    ----------------------
    label (str, optional)
        Label to identify the output files/directories.
    array_config (str, required)
        Path to a yaml file with the array config data.
    verbosity (str, optional)
        Log level to print.

    Example
    -------
    North - Prod5, simple example

    Get the array configuration from DB:

    .. code-block:: console

        simtools-db-get-file-from-db --file_name array_config_test.yml

    Run the application. Runtime < 1 min.

    .. code-block:: console

        simtools-generate-array-config --label test --array_config array_config_test.yml

    The output is saved in simtools-output/test/model.
"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.model.array_model import ArrayModel


def _parse():
    config = configurator.Configurator(
        label=Path(__file__).stem,
        description=("Example of how to generate sim_telarray config files for a given array."),
    )
    config.parser.add_argument(
        "--array_config",
        help="Yaml file with array config data.",
        type=str,
        required=True,
    )
    return config.initialize(db_config=True, simulation_model="version")


def main():  # noqa: D103
    args_dict, db_config = _parse()

    logger = logging.getLogger("simtools")
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    array_config_data = gen.collect_data_from_file_or_dict(args_dict["array_config"], None)

    array_model = ArrayModel(
        label=args_dict["label"],
        model_version=args_dict["model_version"],
        mongo_db_config=db_config,
        site=array_config_data["common"].get("site"),
        layout_name=array_config_data["common"].get("layout_name"),
        parameters_to_change=array_config_data.get("array"),
    )

    # Printing list of telescope for quick inspection.
    array_model.print_telescope_list()

    # Exporting config files.
    array_model.export_all_simtel_config_files()


if __name__ == "__main__":
    main()

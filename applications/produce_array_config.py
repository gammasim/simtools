#!/usr/bin/python3

"""
    Summary
    -------
    This application is an example of how to produce sim_telarray config \
    files for a given array.

    All the input required has to be given as a yaml file by the command \
    line argument array_config.

    The required entries in the array_config file are:

    'site': South or North

    'layoutName': name of a valid layout array.

    'modelVersion': name of a valid model version.

    'default': telescope model names to be assigned to each telescope size by default. \
    It must contain entries for 'LST' and 'MST' (and 'SST' in case of South site).

    As optional data, specific telescope models can be assigned to individual telescopes. \
    This is done by the entries with the name of the telescope (as used by the layout \
    definition, ex. LST-01, MST-05, SST-10).

    Each telescope model can be set in two ways.

    a) A single str with the name of telescope model.
    Ex. 'MST-05': 'NectarCam-D'

    b) A dict containing a 'name' key with the name of the telescope model and further keys \
    with model parameters to be changed from the original model.
    Ex.:

    .. code-block:: python

        'MST-05': {
                'name': 'NectarCam-D',
                'fadc_pulse_shape': 'Pulse_template_nectarCam_17042020-noshift.dat',
                'discriminator_pulse_shape': 'Pulse_template_nectarCam_17042020-noshift.dat'
            }

    This is an example of the content of an array_config file.

    .. code-block:: python

        site: North,
        layoutName: Prod5
        modelVersion: Prod5
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
        Log level to print (default=INFO).

    Example
    -------
    North - Prod5, simple example

    Runtime < 1 min.

    .. code-block:: console

        python applications/produce_array_config.py --label test \
            --array_config data/test-data/arrayConfigTest.yml -v DEBUG

    All the produced model files can be found in simtools-output/test/model/

"""

import logging
from pathlib import Path

import simtools.configuration as configurator
import simtools.util.general as gen
from simtools.model.array_model import ArrayModel


def main():

    config = configurator.Configurator(
        label=Path(__file__).stem,
        description=("Example of how to produce sim_telarray config files for a given array."),
    )
    config.parser.add_argument(
        "--array_config",
        help="Yaml file with array config data.",
        type=str,
        required=True,
    )
    args_dict, db_config = config.initialize(db_config=True)

    logger = logging.getLogger("simtools")
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    arrayModel = ArrayModel(
        label=args_dict["label"],
        mongoDBConfig=db_config,
        arrayConfigFile=args_dict["array_config"],
    )

    # Printing list of telescope for quick inspection.
    arrayModel.print_telescope_list()

    # Exporting config files.
    arrayModel.export_all_simtel_config_files()


if __name__ == "__main__":
    main()

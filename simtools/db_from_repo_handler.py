"""
Module to mimic DB interaction for simulation model DB development.
Read simulation model values from files in git repository.

"""

import logging
from pathlib import Path

import simtools.constants
import simtools.utils.general as gen
from simtools.utils.names import site_parameters

logger = logging.getLogger(__name__)


def update_site_parameters_from_repo(parameters, site, model_version):
    """
    Update site parameters with values from a repository.
    Existing entries will be updated, new entries will be added.
    TODO - model version is ignored at this point (expect that repository
    URL is set to the correct version)

    Parameters
    ----------
    parameters: dict
        Dictionary with parameters to be updated.
    site: str
        South or North.
    model_version: str
        Model version to use.

    Returns
    -------
    dict
        Updated dictionary with parameters.

    """

    if simtools.constants.SIMULATION_MODEL_URL is None:
        logger.debug("No repository specified, skipping site parameter update")
        return parameters

    for key, value in site_parameters.items():
        file_path = Path(simtools.constants.SIMULATION_MODEL_URL, "Site", site, f"{key}.json")
        try:
            logger.debug(
                f"Parameter {key}/{value['name']} updated for {model_version} "
                f"from repository file {file_path}"
            )
            parameters[key] = gen.collect_data_from_file_or_dict(file_name=file_path, in_dict=None)
        except FileNotFoundError:
            logger.debug(
                f"Parameter {value['name']} not updated for {model_version};"
                f"missing repository file {file_path}"
            )
            continue

    return parameters

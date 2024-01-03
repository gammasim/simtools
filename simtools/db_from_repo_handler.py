"""
Module to mimic DB interaction for simulation model DB development.
Read simulation model values from files in git repository.

"""

import logging
from pathlib import Path

import simtools.constants
import simtools.utils.general as gen

logger = logging.getLogger(__name__)


def update_site_parameters_from_repo(parameters, site, model_version):
    """
    Update site parameters with values from repositories.
    Existing entries will be updated, new entries will be added.

    Parameters
    ----------
    parameters: dict
        Dictionary with parameters to be updated.
    site: str
        South or North.

    """

    logger.info(
        f"Updating site parameters from repository for {site} site"
        f" and model version {model_version}"
    )

    for key, value in parameters.items():
        file_path = Path(simtools.constants.SIMULATION_MODEL_URL, "Site", site, f"{key}.json")
        if file_path.exists():
            logger.info(f"Updating parameter {key} for {site} from repository file {file_path}")
            parameters[key] = gen.collect_data_from_yaml_or_dict(in_yaml=file_path, in_dict=None)
            logger.info(f"Old value: {value}, new value: {parameters[key]}")
        else:
            logger.info(f"Parameter {key} for {site} not found in repository (file {file_path})")
            logger.info(f"Keeping old value: {value}")

    return parameters

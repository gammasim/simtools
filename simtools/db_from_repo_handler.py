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
    Update site parameters with values from a repository.
    Existing entries will be updated, new entries will be added.

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

    logger.debug(
        f"Updating site parameters from repository for {site} site"
        f" and model version {model_version}"
    )

    for key, value in site_parameters.items():
        file_path = Path(simtools.constants.SIMULATION_MODEL_URL, "Site", site, f"{value}.json")
        try:
            logger.debug(f"Updating parameter {key} for {site} from repository file {file_path}")
            parameters[key] = gen.collect_data_from_file_or_dict(file_name=file_path, in_dict=None)
        except FileNotFoundError:
            logger.error("Missing parameter file %s", file_path)
            raise
        try:
            logger.debug(f"Updated parameter {key} for {site} from old value: {parameters[key]}")
        except KeyError:
            logger.debug(f"Added parameter {key} for {site}")
        parameters[key] = gen.collect_data_from_file_or_dict(file_name=file_path, in_dict=None)

    return parameters


# simulation_model parameter naming to DB parameter naming mapping
# TODO - decide if this should be moved somewhere else
site_parameters = {
    "altitude": "reference_point_altitude",
    "ref_lon": "reference_point_longitude",
    "ref_lat": "reference_point_latitude",
    "corsika_obs_level": "corsika_observation_level",
}

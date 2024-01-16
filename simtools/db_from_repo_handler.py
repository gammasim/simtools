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
                f"Parameter {key}/{value} updated for {model_version} "
                f"from repository file {file_path}"
            )
            parameters[key] = gen.collect_data_from_file_or_dict(file_name=file_path, in_dict=None)
        except FileNotFoundError:
            logger.debug(
                f"Parameter {value} not updated for {model_version};"
                f"missing repository file {file_path}"
            )
            continue

    return parameters


# simulation_model parameter naming to DB parameter naming mapping
site_parameters = {
    # Note inconsistency between old and new model
    # altitude was the corsika observation level in the old model
    "reference_point_altitude": "altitude",
    "reference_point_longitude": "ref_long",
    "reference_point_latitude": "ref_lat",
    # Note naming inconsistency between old and new model
    # altitude was the corsika observation level in the old model
    "corsika_observation_level": "altitude",
    "epsg_code": "EPSG",
}

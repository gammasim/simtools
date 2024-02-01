"""
Module to mimic DB interaction for simulation model DB development.
Read simulation model values from files in simulation model repository.

"""

import logging

import simtools.utils.general as gen
from simtools.utils import names

logger = logging.getLogger(__name__)


def update_model_parameters_from_repo(
    parameters, site, telescope_model_name, model_version, db_simulation_model_url
):
    """
    Update model parameters with values from a repository.
    Existing entries will be updated, new entries will be added.

    Parameters
    ----------
    parameters: dict
        Dictionary with parameters to be updated.
    site: str
        Observatory site (e.g., South or North)
    telescope_model_name: str
        Name of the telescope model (e.g. LST-1, MST-FlashCam-D)
    model_version: str
        Model version to use.
    db_simulation_model_url: str
        URL to the simulation model repository.

    Returns
    -------
    dict
        Updated dictionary with parameters.

    """

    logger.info("Updating model parameters from repository")

    _array_element_id = names.array_element_id_from_telescope_model_name(
        site=site,
        telescope_model_name=telescope_model_name,
    )
    logger.info("Site: %s, telescope: %s (%s)", site, telescope_model_name, _array_element_id)

    return _update_parameters_from_repo(
        parameters=parameters,
        site=site,
        array_element_id=_array_element_id,
        model_version=model_version,
        parameter_to_query=names.telescope_parameters,
        parameter_collection="telescope",
        db_simulation_model_url=db_simulation_model_url,
    )


def update_site_parameters_from_repo(parameters, site, model_version, db_simulation_model_url):
    """
    Update site parameters with values from a repository.
    Existing entries will be updated, new entries will be added.

    Parameters
    ----------
    parameters: dict
        Dictionary with parameters to be updated.
    site: str
        Observatory site (e.g., South or North)
    model_version: str
        Model version to use.
    db_simulation_model_url: str
        URL to the simulation model repository.

    Returns
    -------
    dict
        Updated dictionary with parameters.

    """

    return _update_parameters_from_repo(
        parameters=parameters,
        site=site,
        array_element_id=None,
        model_version=model_version,
        parameter_to_query=names.site_parameters,
        parameter_collection="site",
        db_simulation_model_url=db_simulation_model_url,
    )


def _update_parameters_from_repo(
    parameters,
    site,
    array_element_id,
    model_version,
    parameter_to_query,
    parameter_collection,
    db_simulation_model_url,
):
    """
    Update parameters with values from a repository.
    Existing entries will be updated, new entries will be added.

    Parameters
    ----------
    parameters: dict
        Dictionary with parameters to be updated.
    site: str
        Observatory site (e.g., South or North)
    array_element_id: str
        Array element ID (e.g., LSTN, or LSTN-01)
    model_version: str
        Model version to use.
    parameter_to_query: dict
        Dictionary with parameter names and labels to be queried.
    parameter_collection: str
        Collection of parameters to be queried (e.g., telescope or site parameters).
    db_simulation_model_url: str
        URL to the simulation model repository.

    Returns
    -------
    dict
        Updated dictionary with parameters.

    """

    if db_simulation_model_url is None:
        logger.debug(f"No repository specified, skipping {parameter_collection} parameter updates")
        return parameters
    logger.warning(f"Ignoring model version {model_version} in parameter updates (TODO)")

    if parameter_collection in ["telescope", "calibration"]:
        _file_path = gen.join_url_or_path(db_simulation_model_url, array_element_id)
        # ID-independent array element name (meaning e.g., MSTN instead of MSTN-03)
        # (required below if there is only one telescope model defined for the class)
        _array_element_without_id = names.get_telescope_class(
            array_element_id, site=names.get_site_from_telescope_name(array_element_id)
        )
        if _array_element_without_id == array_element_id:
            _array_element_without_id = None
    elif parameter_collection == "site":
        _file_path = gen.join_url_or_path(db_simulation_model_url, "Site", site)
        _array_element_without_id = None
    else:
        logger.error(f"Unknown parameter collection {parameter_collection}")
        raise ValueError

    for key in parameter_to_query.keys():
        _parameter_file = gen.join_url_or_path(_file_path, f"{key}.json")
        try:
            parameters[key] = gen.collect_data_from_file_or_dict(
                file_name=_parameter_file, in_dict=None
            )
            logger.debug(f"Parameter {key} updated from repository file {_parameter_file}")
        except (FileNotFoundError, gen.InvalidConfigData):
            # try ID-independent model using _array_element_without_id
            # (meaning e.g., MSTN instead of MSTN-03)
            # accept errors, as not all parameters are defined in the repository
            try:
                _parameter_file = gen.join_url_or_path(
                    db_simulation_model_url, _array_element_without_id, f"{key}.json"
                )
                parameters[key] = gen.collect_data_from_file_or_dict(
                    file_name=_parameter_file, in_dict=None
                )
                logger.debug(f"Parameter {key} updated from repository file {_parameter_file}")
            except (FileNotFoundError, TypeError, gen.InvalidConfigData):
                pass

    return parameters

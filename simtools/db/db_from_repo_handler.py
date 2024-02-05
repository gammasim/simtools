"""
Module to mimic DB interaction for simulation model DB development.
Read simulation model values from files in simulation model repository.

"""

import logging

import simtools.utils.general as gen
from simtools.utils import names

logger = logging.getLogger(__name__)


def update_model_parameters_from_repo(
    parameters, site, telescope_name, model_version, db_simulation_model_url
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
    telescope_name: str
        Telescope name (e.g., MSTN-01, MSTN-DESIGN)
    model_version: str
        Model version to use.
    db_simulation_model_url: str
        URL to the simulation model repository.

    Returns
    -------
    dict
        Updated dictionary with parameters.

    """

    logger.info(
        "Updating model parameters from repository for site: %s, telescope: %s",
        site,
        telescope_name,
    )

    return _update_parameters_from_repo(
        parameters=parameters,
        site=site,
        telescope_name=telescope_name,
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
        telescope_name=None,
        model_version=model_version,
        parameter_to_query=names.site_parameters,
        parameter_collection="site",
        db_simulation_model_url=db_simulation_model_url,
    )


def _update_parameters_from_repo(
    parameters,
    site,
    telescope_name,
    model_version,
    parameter_to_query,
    parameter_collection,
    db_simulation_model_url,
    db_simulation_model="verified_model",
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
    telescope_name: str
        Telescope name (e.g., MSTN-01, MSTN-DESIGN)
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
        _file_path = gen.join_url_or_path(
            db_simulation_model_url, db_simulation_model, telescope_name
        )
        # use design telescope model in case there is no model defined for this telescope ID
        _design_model = names.get_telescope_type_from_telescope_name(telescope_name) + "-design"
        if _design_model == telescope_name:
            _design_model = None
    elif parameter_collection == "site":
        _file_path = gen.join_url_or_path(
            db_simulation_model_url, db_simulation_model, "Site", site
        )
        _design_model = None
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
            # use design telescope model in case there is no model defined for this telescope ID
            # accept errors, as not all parameters are defined in the repository
            try:
                _parameter_file = gen.join_url_or_path(
                    db_simulation_model_url, db_simulation_model, _design_model, f"{key}.json"
                )
                parameters[key] = gen.collect_data_from_file_or_dict(
                    file_name=_parameter_file, in_dict=None
                )
                logger.debug(f"Parameter {key} updated from repository file {_parameter_file}")
            except (FileNotFoundError, TypeError, gen.InvalidConfigData):
                pass

    return parameters


def get_list_of_model_parameters(model_type, db_simulation_model_url):
    """
    Get list of telescope model parameters from repository.

    Parameters
    ----------
    model_type: str
        Simulation model type (e.g., telescope, site)
    db_simulation_model_url: str
        URL to the simulation model repository.

    Returns
    -------
    list
        List of telescope model parameters.

    """

    _pars = gen.collect_data_from_file_or_dict(
        file_name=gen.join_url_or_path(
            db_simulation_model_url, "parameter_lists", f"{model_type}.list"
        ),
        in_dict=None,
    )

    return _pars
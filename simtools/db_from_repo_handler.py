"""
Module to mimic DB interaction for simulation model DB development.
Read simulation model values from files in git repository.

"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.utils import names

logger = logging.getLogger(__name__)


def update_model_parameters_from_repo(
    parameters, site, telescope_model_name, model_version, db_simulation_model_url
):
    """
    Update model parameters with values from a repository.
    Existing entries will be updated, new entries will be added.
    TODO - model version is ignored at this point (expect that repository
    URL is set to the correct version)

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
    logger.info("Site: %s, telescope: %s", site, telescope_model_name)

    _id = names.array_element_id_from_telescope_model_name(
        site=site,
        telescope_model_name=telescope_model_name,
    )

    return _update_parameters_from_repo(
        parameters=parameters,
        site=site,
        array_element_id=_id,
        model_version=model_version,
        parameter_to_query=names.telescope_parameters,
        db_simulation_model_url=db_simulation_model_url,
    )


def update_site_parameters_from_repo(parameters, site, model_version, db_simulation_model_url):
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
        db_simulation_model_url=db_simulation_model_url,
    )


def _update_parameters_from_repo(
    parameters, site, array_element_id, model_version, parameter_to_query, db_simulation_model_url
):
    """
    Update parameters with values from a repository.
    Existing entries will be updated, new entries will be added.
    The database collections are identified by:
    - array_element_id not None: telescope parameters
    - site not None and array_element_id None: site parameters

    TODO - model version is ignored at this point (expect that repository
    URL is set to the correct version)

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
    db_simulation_model_url: str
        URL to the simulation model repository.

    Returns
    -------
    dict
        Updated dictionary with parameters.

    """

    if db_simulation_model_url is None:
        logger.debug("No repository specified, skipping site parameter update")
        return parameters

    # telescope model
    if array_element_id is not None:
        file_path = Path(
            db_simulation_model_url,
            array_element_id,
        )
        # no ID-dependent model (meaning none for e.g., MSTN-03, but for MSTN)
        if not file_path.exists():
            file_path = Path(
                db_simulation_model_url,
                names.get_telescope_class(
                    array_element_id,
                    site=names.get_site_from_telescope_name(array_element_id),
                ),
            )
    # site model
    else:
        file_path = Path(db_simulation_model_url, "Site", site)
    if not file_path.exists():
        logger.debug(
            f"No repository found, skipping site parameter update (searched in {file_path})"
        )
        return parameters

    logger.debug("Reading parameters from %s", file_path)

    for key, value in parameter_to_query.items():
        _parameter_file = file_path / f"{key}.json"
        try:
            logger.debug(
                f"Parameter {key}/{value['name']} updated for {model_version} "
                f"from repository file {_parameter_file}"
            )
            parameters[key] = gen.collect_data_from_file_or_dict(
                file_name=_parameter_file, in_dict=None
            )
            logger.debug(f"Parameter {key} updated to {parameters[key]}")
        except FileNotFoundError:
            # TODO
            # expected for now, as not all parameters are defined in the simulation_model repository
            continue

    return parameters

"""
Module to mimic DB interaction for simulation model DB development.

Read simulation model values from files in simulation model repository.

"""

import logging

import simtools.utils.general as gen
from simtools.utils import names

logger = logging.getLogger(__name__)


def update_model_parameters_from_repo(
    parameters,
    site,
    array_element_name,
    parameter_collection,
    model_version,
    db_simulation_model_url,
):
    """
    Update model parameters with values from a repository.

    Existing entries will be updated, new entries will be added.

    Parameters
    ----------
    parameters: dict
        Existing dictionary with parameters to be updated.
    site: str
        Observatory site (e.g., South or North)
    array_element_name: str
        Array element name (e.g., MSTN-01, MSTN-design)
    parameter_collection: str
        Collection of parameters to be queried (e.g., telescope or site)
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
        "Updating model parameters from repository for site: %s, array element: %s",
        site,
        array_element_name,
    )

    if db_simulation_model_url is None:
        logger.debug(f"No repository specified, skipping {parameter_collection} parameter updates")
        return parameters

    if parameter_collection in ["telescopes", "calibration"]:
        _file_path = gen.join_url_or_path(
            db_simulation_model_url,
            "model_versions",
            model_version,
            array_element_name,
        )
        # use design array element model in case there is no model defined for this array element ID
        _design_model = names.get_array_element_type_from_name(array_element_name) + "-design"
        if _design_model == array_element_name:
            _design_model = None
    elif parameter_collection == "site":
        _file_path = gen.join_url_or_path(
            db_simulation_model_url,
            "model_versions",
            model_version,
            "OBS-" + site,
        )
        _design_model = None
    else:
        raise ValueError(f"Unknown parameter collection {parameter_collection}")

    for key in parameters:
        _tmp_par = {}
        _parameter_file = gen.join_url_or_path(_file_path, f"{key}.json")
        try:
            _tmp_par = gen.collect_data_from_file_or_dict(file_name=_parameter_file, in_dict=None)
        except (FileNotFoundError, gen.InvalidConfigDataError):
            # use design array element model in case there is no model defined for this
            #  array element ID. Accept errors, as not all parameters are defined in the repository
            try:
                _file_path = gen.join_url_or_path(
                    db_simulation_model_url,
                    "model_versions",
                    model_version,
                    _design_model,
                )
                _tmp_par = gen.collect_data_from_file_or_dict(
                    file_name=gen.join_url_or_path(_file_path, f"{key}.json"), in_dict=None
                )
            except (FileNotFoundError, TypeError, gen.InvalidConfigDataError):
                pass
        if _tmp_par.get("version") == model_version:
            parameters[key] = _tmp_par

    # return all entries which are not None
    return {key: value for key, value in parameters.items() if value is not None}

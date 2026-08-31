"""Build catalog metadata for completed simulation-production jobs."""

from astropy import units as u

from simtools.utils import names

CATALOG_SITE_NAMES = {"North": "LaPalma", "South": "Paranal"}


def build_simulation_job_metadata(args_dict, simulator):
    """Build DIRAC catalog metadata from resolved simulation configuration.

    Parameters
    ----------
    args_dict : dict
        Resolved ``simulate_prod`` application arguments.
    simulator : simtools.simulator.Simulator
        Simulator for the completed run.

    Returns
    -------
    dict
        Job-level metadata using DIRAC file-catalog field names.
    """
    azimuth_angle = args_dict["azimuth_angle"].to_value(u.deg)
    view_cone_min, view_cone_max = args_dict["view_cone"]
    metadata = {
        "array_layout": args_dict["array_layout_name"],
        "site": CATALOG_SITE_NAMES[args_dict["site"]],
        "particle": args_dict["primary"].lower(),
        "phiP": round((azimuth_angle + 180.0) % 360.0, 2),
        "thetaP": round(float(args_dict["zenith_angle"].to_value(u.deg)), 2),
        "sct": str(_has_sct(simulator.array_models)),
        "view_cone_min": round(float(view_cone_min.to_value(u.deg)), 2),
        "view_cone_max": round(float(view_cone_max.to_value(u.deg)), 2),
        "runNumber": int(simulator.run_number),
        "model_version": str(args_dict["model_version"]),
    }
    _add_optional_coordinate(metadata, "dec", args_dict.get("dec"))
    _add_optional_coordinate(metadata, "ha", args_dict.get("ha"))
    return metadata


def _has_sct(array_models):
    """Return whether any resolved array model contains an SCT."""
    return any(
        names.get_array_element_type_from_name(element_name) == "SCTS"
        for array_model in array_models
        for element_name in array_model.array_elements
    )


def _add_optional_coordinate(metadata, key, value):
    """Add one optional angular coordinate in degrees to metadata."""
    if value is not None:
        metadata[key] = round(float(value.to_value(u.deg)), 2)

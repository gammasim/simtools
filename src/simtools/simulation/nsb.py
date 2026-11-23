"""Night-sky background rate scaling.

The NSB rates written to sim_telarray configuration files are scaled here, so that the
configuration and the output validation use the same expected values.
"""

import math
from collections import namedtuple
from copy import deepcopy

# NSB settings used for an export: array zenith angle in degrees, manual rate factor
# (None: site value), and whether the airmass scaling is applied.
NsbSettings = namedtuple("NsbSettings", ["zenith_angle", "manual_factor", "autoscale"])

_NSB_PIXEL_RATE = "nsb_pixel_rate"
_NSB_AUTOSCALE_AIRMASS = "nsb_autoscale_airmass"
_NSB_SCALING_FACTOR = "nsb_scaling_factor"
_AUTOSCALE_AIRMASS_DEFAULT = [0.0, 0.15]


def airmass_scale(zenith_angle, airglow_fraction, extinction):
    """Calculate the NSB rate relative to zenith.

    See Garstang 1989, PASP, 101, 306 for the airmass scaling of the night-sky background.

    Parameters
    ----------
    zenith_angle : float
        Array zenith angle in degrees.
    airglow_fraction : float
        Fraction of the zenith NSB attributed to airglow.
    extinction : float
        Dimensionless effective extinction coefficient.

    Returns
    -------
    float
        Dimensionless rate scaling factor.
    """
    zenith = math.radians(zenith_angle)
    airmass = (1.0 - 0.96 * math.sin(zenith) ** 2) ** -0.5
    return (1.0 - airglow_fraction + airglow_fraction * airmass) * math.exp(
        -extinction * (airmass - 1.0)
    )


def site_nsb_factor(site_parameters):
    """Return the site NSB scaling factor from resolved site parameters (1.0 if not set).

    Parameters
    ----------
    site_parameters : dict
        Site parameters including sim_telarray software settings (see
        ``get_simtel_site_parameters``).

    Returns
    -------
    float
        Site NSB scaling factor.
    """
    return float(site_parameters.get(_NSB_SCALING_FACTOR, {"value": 1.0})["value"])


def scale_nsb_parameters(parameters, settings, site_factor):
    """Return a copy of model parameters with the NSB values sim_telarray is given.

    NSB pixel rates are multiplied by the scaling factor (airmass scaling if applicable,
    otherwise the manual or site factor). The airmass scaling is then applied by simtools,
    so the sim_telarray airmass parameter keeps only the extinction and the scaling factor
    is set to 1.0. The input parameters are not modified.

    Parameters
    ----------
    parameters : dict
        Model parameters (name -> {"value": ...}), values may be strings.
    settings : NsbSettings
        NSB settings used for the export.
    site_factor : float
        Site NSB scaling factor.

    Returns
    -------
    dict
        Copy of the parameters with NSB values as seen by sim_telarray.
    """
    result = deepcopy(parameters)
    auto = _parse_value(
        parameters.get(_NSB_AUTOSCALE_AIRMASS, {}).get("value", _AUTOSCALE_AIRMASS_DEFAULT)
    )
    if _NSB_PIXEL_RATE in result:
        factor = _rate_factor(auto, settings, site_factor)
        rate = _parse_value(result[_NSB_PIXEL_RATE]["value"])
        if isinstance(rate, list):
            result[_NSB_PIXEL_RATE]["value"] = [value * factor for value in rate]
        else:
            result[_NSB_PIXEL_RATE]["value"] = rate * factor
    if _NSB_AUTOSCALE_AIRMASS in result:
        result[_NSB_AUTOSCALE_AIRMASS]["value"] = [0.0, auto[1]]
    if _NSB_SCALING_FACTOR in result:
        result[_NSB_SCALING_FACTOR]["value"] = 1.0
    return result


def _rate_factor(auto, settings, site_factor):
    """Return the NSB rate factor for the given settings."""
    factor = settings.manual_factor if settings.manual_factor is not None else site_factor
    if (
        settings.autoscale
        and math.isclose(factor, 1.0, rel_tol=0.0, abs_tol=0.0)
        and not math.isclose(auto[0], 0.0, rel_tol=0.0, abs_tol=0.0)
    ):
        factor = airmass_scale(settings.zenith_angle, auto[0], auto[1])
    return factor


def _parse_value(value):
    """Decode numeric scalars and lists stored as strings in the model repository."""
    if not isinstance(value, str):
        return value
    values = [float(item) for item in value.replace(",", " ").split()]
    return values[0] if len(values) == 1 else values

"""Night-sky background rate scaling."""

import math


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

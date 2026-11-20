"""Default values for CLI and settings configuration."""

from simtools import constants
from simtools.runtime import software_catalog

CORSIKA_HE_INTERACTION = constants.CORSIKA_HE_INTERACTION
CORSIKA_LE_INTERACTION = constants.CORSIKA_LE_INTERACTION
ZENITH_ANGLE_SCALING_FACTOR_DEFAULT = constants.ZENITH_ANGLE_SCALING_FACTOR_DEFAULT

CORSIKA_INTERACTION_TABLE_PATH = constants.CORSIKA_INTERACTION_TABLE_PATH

# Minimum zenith angle (degrees) above which CORSIKA uses a curved-atmosphere binary.
CURVED_ATMOSPHERE_MIN_ZENITH_ANGLE_DEG = 65

# Simulation software identifiers (choices) and the default simulation chain.
SIMULATION_SOFTWARE_CHOICES = constants.SIMULATION_SOFTWARE_CHOICES
SIMULATION_SOFTWARE_DEFAULT = software_catalog.default_chain(software_catalog.AIR_SHOWER_SOURCE)

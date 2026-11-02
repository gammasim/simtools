"""Default values for CLI and settings configuration."""

from simtools import constants

CORSIKA_HE_INTERACTION = constants.CORSIKA_HE_INTERACTION
CORSIKA_LE_INTERACTION = constants.CORSIKA_LE_INTERACTION
ZENITH_ANGLE_SCALING_FACTOR_DEFAULT = constants.ZENITH_ANGLE_SCALING_FACTOR_DEFAULT

CORSIKA_INTERACTION_TABLE_PATH = (
    "/workdir/external/simpipe/simulation_software/corsika7-interaction-tables/interaction-tables/"
)

# Minimum zenith angle (degrees) above which CORSIKA uses a curved-atmosphere binary.
CURVED_ATMOSPHERE_MIN_ZENITH_ANGLE_DEG = 65

# Simulation software identifiers and the external executables each requires.
SIMULATION_SOFTWARE_DEPENDENCIES = {
    "corsika": frozenset({"corsika"}),
    "sim_telarray": frozenset({"sim_telarray"}),
    "corsika_sim_telarray": frozenset({"corsika", "sim_telarray"}),
}
SIMULATION_SOFTWARE_CHOICES = tuple(SIMULATION_SOFTWARE_DEPENDENCIES)
SIMULATION_SOFTWARE_DEFAULT = "corsika_sim_telarray"

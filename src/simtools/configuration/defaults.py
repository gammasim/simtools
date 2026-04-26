"""Default values for CLI and settings configuration."""

CORSIKA_PATH = "/workdir/simulation_software/corsika7"
CORSIKA_INTERACTION_TABLE_PATH = (
    "/workdir/external/simpipe/simulation_software/corsika7-interaction-tables/interaction-tables/"
)
CORSIKA_HE_INTERACTION = "epos"
CORSIKA_LE_INTERACTION = "urqmd"

# Minimum zenith angle (degrees) above which CORSIKA uses a curved-atmosphere binary.
CURVED_ATMOSPHERE_MIN_ZENITH_ANGLE_DEG = 65

# Valid simulation software identifiers and the default choice.
SIMULATION_SOFTWARE_CHOICES = ("corsika", "sim_telarray", "corsika_sim_telarray")
SIMULATION_SOFTWARE_DEFAULT = "corsika_sim_telarray"

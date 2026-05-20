import simtools.configuration.defaults as defaults


def test_corsika_defaults():
    """Test default CORSIKA configuration constants."""
    assert defaults.CORSIKA_PATH == "/workdir/simulation_software/corsika7"
    assert (
        defaults.CORSIKA_INTERACTION_TABLE_PATH
        == "/workdir/external/simpipe/simulation_software/corsika7-interaction-tables/interaction-tables/"
    )
    assert defaults.CORSIKA_HE_INTERACTION == "epos"
    assert defaults.CORSIKA_LE_INTERACTION == "urqmd"


def test_simulation_software_defaults():
    """Test simulation software default values."""
    assert defaults.CURVED_ATMOSPHERE_MIN_ZENITH_ANGLE_DEG == 65
    assert defaults.SIMULATION_SOFTWARE_CHOICES == (
        "corsika",
        "sim_telarray",
        "corsika_sim_telarray",
    )
    assert defaults.SIMULATION_SOFTWARE_DEFAULT == "corsika_sim_telarray"

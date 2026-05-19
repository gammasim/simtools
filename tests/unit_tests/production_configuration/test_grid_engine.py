from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from simtools.production_configuration.observation_grid import ProductionGridEngine


def test_generate_simulation_grid_keeps_horizontal_coordinates_for_radec_axes():
    axes = {
        "axes": {
            "ra": {"range": [0, 0], "binning": 1, "scaling": "linear", "units": "deg"},
            "dec": {"range": [0, 0], "binning": 1, "scaling": "linear", "units": "deg"},
        }
    }
    engine = ProductionGridEngine(
        axes=axes,
        coordinate_system="ra_dec",
        observing_location=EarthLocation(lat=28.76 * u.deg, lon=-17.89 * u.deg, height=2200 * u.m),
        observing_time=Time("2017-09-16 00:00:00", scale="utc"),
        lookup_table=None,
    )

    simulation_grid = engine.generate_simulation_grid()

    assert "zenith_angle" in simulation_grid[0]
    assert "azimuth" in simulation_grid[0]
    assert "ra" in simulation_grid[0]
    assert "dec" in simulation_grid[0]

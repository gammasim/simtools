import astropy.units as u
import pytest

from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.simtel_light_emission import SimulatorLightEmission


@pytest.fixture
def label():
    return "test-simtel-light-emission"


@pytest.fixture
def default_config():
    return {
        "beam_shape": {
            "len": 1,
            "unit": str,
            "default": "Gauss",
            "names": ["beam_shape", "angular_distribution"],
        },
        "beam_width": {
            "len": 1,
            "unit": u.Unit("deg"),
            "default": 0.5 * u.deg,
            "names": ["rms"],
        },
        "pulse_shape": {
            "len": 1,
            "unit": str,
            "default": "Gauss",
            "names": ["pulse_shape"],
        },
        "pulse_width": {
            "len": 1,
            "unit": u.Unit("deg"),
            "default": 5 * u.ns,
            "names": ["rms"],
        },
        "x_pos": {
            "len": 1,
            "unit": u.Unit("cm"),
            "default": 0 * u.cm,
            "names": ["x_position"],
        },
        "y_pos": {
            "len": 1,
            "unit": u.Unit("cm"),
            "default": 0 * u.m,
            "names": ["y_position"],
        },
        "z_pos": {
            "len": 1,
            "unit": u.Unit("cm"),
            "default": [i * 100 for i in [200, 300, 400, 600, 800, 1200, 2000, 4000]] * u.cm,
            "names": ["z_position"],
        },
        "direction": {
            "len": 3,
            "unit": u.dimensionless_unscaled,
            "default": [0, 0.0, -1],
            "names": ["direction", "cx,cy,cz"],
        },
    }


@pytest.fixture
def simulator(db_config, default_config, label, simtel_path_no_mock, io_handler):
    version = "Released"
    simtel_source_path = simtel_path_no_mock
    tel = TelescopeModel(
        site="North",
        telescope_model_name="LST-1",
        model_version=version,
        label="test",
        mongo_db_config=db_config,
    )
    # default_le_config = default_config
    le_application = "xyzls"
    simulator = SimulatorLightEmission(
        tel, default_config, le_application, label, simtel_source_path, config_data={}
    )
    return simulator


def test_initialization(simulator, default_config):
    assert isinstance(simulator, SimulatorLightEmission)
    assert simulator.le_application == "xyzls"
    assert (
        simulator.default_le_config == default_config
    )  # Update with the expected default configuration


def test_runs(simulator):
    assert simulator.runs == 1


def test_photons_per_run_default(simulator):
    assert simulator.photons_per_run == 1e10

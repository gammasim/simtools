import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

from simtools.model.calibration_model import CalibrationModel
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.simulator_light_emission import SimulatorLightEmission
from simtools.utils import general as gen
from simtools.visualization.visualize import plot_simtel_ctapipe


@pytest.fixture(name="label")
def label_fixture():
    return "test-simtel-light-emission"


@pytest.fixture(name="default_config")
def default_config_fixture():
    return {
        "x_pos": {
            "len": 1,
            "unit": u.Unit("cm"),
            "default": 0 * u.cm,
            "names": ["x_position"],
        },
        "y_pos": {
            "len": 1,
            "unit": u.Unit("cm"),
            "default": 0 * u.cm,
            "names": ["y_position"],
        },
        "z_pos": {
            "len": 1,
            "unit": u.Unit("cm"),
            "default": 100000 * u.cm,
            "names": ["z_position"],
        },
        "direction": {
            "len": 3,
            "unit": u.dimensionless_unscaled,
            "default": [0, 0, -1],
            "names": ["direction", "cx,cy,cz"],
        },
    }


@pytest.fixture
def mock_simulator(
    db_config, default_config, label, model_version, simtel_path, site_model_north, io_handler
):
    telescope_model = TelescopeModel(
        site="North",
        telescope_name="LSTN-01",
        model_version=model_version,
        label="test-simtel-light-emission",
        mongo_db_config=db_config,
    )
    calibration_model = CalibrationModel(
        site="North",
        calibration_device_model_name="ILLN-01",
        mongo_db_config=db_config,
        model_version=model_version,
        label="test-simtel-light-emission",
    )

    le_application = "xyzls", "layout"
    light_source_type = "led"
    return SimulatorLightEmission(
        telescope_model=telescope_model,
        calibration_model=calibration_model,
        site_model=site_model_north,
        default_le_config=default_config,
        le_application=le_application,
        simtel_path=simtel_path,
        light_source_type=light_source_type,
        label=label,
        test=True,
    )


@pytest.fixture
def mock_simulator_variable(
    db_config, default_config, label, model_version, simtel_path, site_model_north, io_handler
):
    telescope_model = TelescopeModel(
        site="North",
        telescope_name="LSTN-01",
        model_version=model_version,
        label="test-simtel-light-emission",
        mongo_db_config=db_config,
    )
    calibration_model = CalibrationModel(
        site="North",
        calibration_device_model_name="ILLN-01",
        mongo_db_config=db_config,
        model_version=model_version,
        label="test-simtel-light-emission",
    )

    le_application = "xyzls", "variable"
    light_source_type = "led"
    return SimulatorLightEmission(
        telescope_model=telescope_model,
        calibration_model=calibration_model,
        site_model=site_model_north,
        default_le_config=default_config,
        le_application=le_application,
        simtel_path=simtel_path,
        light_source_type=light_source_type,
        label=label,
        test=True,
    )


@pytest.fixture
def mock_simulator_laser(
    db_config, default_config, label, model_version, simtel_path, site_model_north, io_handler
):
    telescope_model = TelescopeModel(
        site="North",
        telescope_name="LSTN-01",
        model_version=model_version,
        label="test-simtel-light-emission",
        mongo_db_config=db_config,
    )
    calibration_model = CalibrationModel(
        site="North",
        calibration_device_model_name="ILLN-01",
        mongo_db_config=db_config,
        model_version=model_version,
        label="test-simtel-light-emission",
    )

    le_application = "ls-beam", "layout"
    light_source_type = "laser"
    return SimulatorLightEmission(
        telescope_model=telescope_model,
        calibration_model=calibration_model,
        site_model=site_model_north,
        default_le_config=default_config,
        le_application=le_application,
        simtel_path=simtel_path,
        light_source_type=light_source_type,
        label=label,
        test=True,
    )


@pytest.fixture
def mock_output_path(label, io_handler):
    return io_handler.get_output_directory(label)


@pytest.fixture
def calibration_model_illn(db_config, io_handler, model_version):
    return CalibrationModel(
        site="North",
        calibration_device_model_name="ILLN-01",
        mongo_db_config=db_config,
        model_version=model_version,
        label="test-simtel-light-emission",
    )


def test_initialization(mock_simulator, default_config):
    assert isinstance(mock_simulator, SimulatorLightEmission)
    assert mock_simulator.le_application[0] == "xyzls"
    assert mock_simulator.light_source_type == "led"
    assert mock_simulator.default_le_config == default_config


def test_runs(mock_simulator):
    assert mock_simulator.runs == 1


def test_photons_per_run_default(mock_simulator):
    assert mock_simulator.photons_per_run == pytest.approx(1e7)


def test_make_light_emission_script(
    mock_simulator,
    telescope_model_lst,
    simtel_path,
    mock_output_path,
    io_handler,
):
    """layout coordinate vector between LST and ILLN"""
    expected_command = (
        f" rm {mock_output_path}/xyzls_layout.simtel.gz\n"
        f"sim_telarray/LightEmission/xyzls"
        " -x -51627.0"
        " -y 5510.0"
        " -z 9200.0"
        " -d 0.979101,-0.104497,-0.174477"
        " -n 10000000.0"
        " -s 300"
        " -p Gauss:0.0"
        " -a isotropic"
        f" -A {mock_output_path}/model/"
        f"{telescope_model_lst.get_parameter_value('atmospheric_profile')}"
        f" -o {mock_output_path}/xyzls.iact.gz\n"
    )

    command = mock_simulator._make_light_emission_script()

    assert command == expected_command


def test_make_light_emission_script_variable(
    mock_simulator_variable,
    telescope_model_lst,
    simtel_path,
    mock_output_path,
    io_handler,
):
    """layout coordinate vector between LST and ILLN"""
    expected_command = (
        f" rm {mock_output_path}/xyzls_variable.simtel.gz\n"
        f"sim_telarray/LightEmission/xyzls"
        " -x 0.0"
        " -y 0.0"
        " -z 100000.0"
        " -d 0,0,-1"
        " -n 10000000.0"
        f" -A {mock_output_path}/model/"
        f"{telescope_model_lst.get_parameter_value('atmospheric_profile')}"
        f" -o {mock_output_path}/xyzls.iact.gz\n"
    )
    assert mock_simulator_variable.le_application[0] == "xyzls"
    assert mock_simulator_variable.le_application[1] == "variable"

    command = mock_simulator_variable._make_light_emission_script()

    assert command == expected_command


def test_make_light_emission_script_laser(
    mock_simulator_laser,
    telescope_model_lst,
    simtel_path,
    mock_output_path,
    io_handler,
):
    # Call the method under test
    command = mock_simulator_laser._make_light_emission_script()

    # Define the expected command
    expected_command = (
        f" rm {mock_output_path}/ls-beam_layout.simtel.gz\n"
        f"sim_telarray/LightEmission/ls-beam"
        " --events 1"
        " --bunches 2500000"
        " --step 0.1"
        " --bunchsize 1"
        " --spectrum "
        f"{mock_simulator_laser._calibration_model.get_parameter_value('laser_wavelength')}"
        " --lightpulse Gauss:"
        f"{mock_simulator_laser._calibration_model.get_parameter_value('laser_pulse_sigtime')}"
        " --laser-position '-51627.0,5510.0,9200.0'"
        " --telescope-theta 79.951773"
        " --telescope-phi 186.091952"
        " --laser-theta 10.048226999999997"
        " --laser-phi 173.908048"
        f" --atmosphere {mock_output_path}/model/"
        f"{mock_simulator_laser._calibration_model.get_parameter_value('atmospheric_profile')}"
        f" -o {mock_output_path}/ls-beam.iact.gz\n"
    )

    # Assert that the command matches the expected command
    assert command == expected_command


def test_calibration_pointing_direction(mock_simulator):
    # Calling calibration_pointing_direction method
    pointing_vector, angles = mock_simulator.calibration_pointing_direction()

    # Expected pointing vector and angles
    expected_pointing_vector = [0.979, -0.104, -0.174]
    expected_angles = [79.952, 186.092, 79.952, 173.908]

    # Asserting expected pointing vector and angles
    np.testing.assert_array_almost_equal(pointing_vector, expected_pointing_vector, decimal=3)
    np.testing.assert_array_almost_equal(angles, expected_angles, decimal=3)


def test_create_postscript(mock_simulator, simtel_path, mock_output_path):
    expected_command = (
        f"hessioxxx/bin/read_cta"
        " --min-tel 1 --min-trg-tel 1"
        " -q --integration-scheme 4 --integration-window 7,3 -r 5"
        " --plot-with-sum-only"
        " --plot-with-pixel-amp --plot-with-pixel-id"
        f" -p {mock_output_path}/postscripts/xyzls_layout_d_1000.ps"
        f" {mock_output_path}/xyzls_layout.simtel.gz\n"
    )

    command = mock_simulator._create_postscript(integration_window=["7", "3"], level="5")

    assert command == expected_command


def test_plot_simtel_ctapipe(mock_simulator, mock_output_path):
    mock_simulator.output_directory = "./tests/resources/"
    cleaning_args = [5, 3, 2]
    filename = f"{mock_simulator.output_directory}/"
    filename += f"{mock_simulator.le_application[0]}_{mock_simulator.le_application[1]}.simtel.gz"
    distance = 1000 * u.m
    fig = plot_simtel_ctapipe(
        filename, cleaning_args=cleaning_args, distance=distance, return_cleaned=True
    )
    assert isinstance(fig, plt.Figure)  # Check if fig is an instance of matplotlib figure


def test_make_simtel_script(mock_simulator):
    mock_file_content = "Sample content of config file"

    # Patch the built-in open function to return a mock file
    with patch("builtins.open", mock_open(read_data=mock_file_content)) as mock_file:
        mock_file.reset_mock()

        # Mock the necessary attributes and methods used within _make_simtel_script
        mock_simulator._simtel_path = MagicMock()
        mock_simulator._telescope_model = MagicMock()
        mock_simulator._site_model = MagicMock()

        mock_simulator._simtel_path.joinpath.return_value = (
            "/path/to/sim_telarray/bin/sim_telarray/"
        )
        path_to_config_directory = "/path/to/config/"
        mock_simulator._telescope_model.config_file_directory = path_to_config_directory
        path_to_config_file = "/path/to/config/config.cfg"
        mock_simulator._telescope_model.get_config_file.return_value = path_to_config_file

        def get_telescope_model_param(param):
            if param == "atmospheric_transmission":
                return "atm_test"
            if param == "array_element_position_ground":
                return (1, 1, 1)
            return MagicMock()

        mock_simulator._telescope_model.get_parameter_value.side_effect = get_telescope_model_param

        mock_simulator._site_model.get_parameter_value.side_effect = lambda param: (
            "999" if param == "corsika_observation_level" else MagicMock()
        )

        mock_simulator.output_directory = "/directory"

        expected_command = (
            "SIM_TELARRAY_CONFIG_PATH='' "
            "/path/to/sim_telarray/bin/sim_telarray/ "
            "-I -I/path/to/config/ "
            "-c /path/to/config/config.cfg "
            "-DNUM_TELESCOPES=1 "
            "-C altitude=999 -C atmospheric_transmission=atm_test "
            "-C TRIGGER_TELESCOPES=1 "
            "-C TELTRIG_MIN_SIGSUM=2 -C PULSE_ANALYSIS=-30 "
            "-C MAXIMUM_TELESCOPES=1 "
            "-C telescope_theta=76.980826 -C telescope_phi=180.17047 "
            "-C power_law=2.68 -C input_file=/directory/xyzls.iact.gz "
            "-C output_file=/directory/xyzls_layout.simtel.gz "
            "-C histogram_file=/directory/xyzls_layout.ctsim.hdata\n"
        )

        command = mock_simulator._make_simtel_script()

        assert command == expected_command


@patch("os.system")
@patch(
    "simtools.simtel.simulator_light_emission.SimulatorLightEmission._make_light_emission_script"
)
@patch("simtools.simtel.simulator_light_emission.SimulatorLightEmission._make_simtel_script")
@patch("builtins.open", create=True)
def test_prepare_script(
    mock_open,
    mock_make_simtel_script,
    mock_make_light_emission_script,
    mock_os_system,
    mock_simulator,
):
    # Mocking data and behavior
    mock_file = Mock()
    mock_open.return_value.__enter__.return_value = mock_file
    mock_make_light_emission_script.return_value = "light_emission_script_command"
    mock_make_simtel_script.return_value = "simtel_script_command"

    # Execute the method
    script_path = mock_simulator.prepare_script(
        generate_postscript=True, integration_window=["7", "3"], level="5"
    )

    # Assertions
    assert gen.program_is_executable(script_path)
    mock_make_light_emission_script.assert_called_once()  # Ensure this mock is called
    mock_make_simtel_script.assert_called_once()  # Ensure this mock is called

    # Check file write calls
    expected_calls = [
        "#!/usr/bin/env bash\n\n",
        "light_emission_script_command\n\n",
        "simtel_script_command\n\n",
        "# Generate postscript\n\n",
        "postscript_command\n\n",
        "# End\n\n",
    ]
    for call_args, expected_content in zip(mock_file.write.call_args_list, expected_calls):
        assert call_args[0][0] == expected_content


def test_remove_line_from_config(mock_simulator):
    # Create a temporary config file with some lines
    config_content = """array_triggers: value1
    axes_offsets: value2
    some_other_config: value3
    """
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        config_path = Path(tmp_file.name)
        tmp_file.write(config_content.encode("utf-8"))

    mock_simulator._remove_line_from_config(config_path, "array_triggers")

    with open(config_path, encoding="utf-8") as f:
        updated_content = f.read()

    expected_content = """
    axes_offsets: value2
    some_other_config: value3
    """
    assert updated_content.strip() == expected_content.strip()

    config_path.unlink()

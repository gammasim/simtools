from unittest.mock import MagicMock, Mock, call, mock_open, patch

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

from simtools.layout.array_layout import ArrayLayout
from simtools.model.calibration_model import CalibrationModel
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.simtel_light_emission import SimulatorLightEmission


@pytest.fixture
def label():
    return "test-simtel-light-emission"


@pytest.fixture
def default_config():
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
        "x_pos_ILLN-01": {
            "len": 1,
            "unit": u.Unit("m"),
            "default": -58718 * u.cm,
            "names": ["x_position"],
        },
        "y_pos_ILLN-01": {
            "len": 1,
            "unit": u.Unit("m"),
            "default": 275 * u.cm,
            "names": ["y_position"],
        },
        "z_pos_ILLN-01": {
            "len": 1,
            "unit": u.Unit("m"),
            "default": 229500 * u.cm,
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
    simtel_source_path = simtel_path
    telescope_model = TelescopeModel(
        site="North",
        telescope_model_name="LSTN-01",
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

    # default_le_config = default_config
    le_application = "xyzls", "layout"
    light_source_type = "led"
    mock_simulator = SimulatorLightEmission(
        telescope_model=telescope_model,
        calibration_model=calibration_model,
        site_model=site_model_north,
        default_config=default_config,
        le_application=le_application,
        simtel_source_path=simtel_source_path,
        light_source_type=light_source_type,
        label=label,
        config_data={},
    )
    return mock_simulator


@pytest.fixture
def mock_simulator_variable(
    db_config, default_config, label, model_version, simtel_path, site_model_north, io_handler
):
    simtel_source_path = simtel_path
    telescope_model = TelescopeModel(
        site="North",
        telescope_model_name="LSTN-01",
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

    # default_le_config = default_config
    le_application = "xyzls", "variable"
    light_source_type = "led"
    mock_simulator_variable = SimulatorLightEmission(
        telescope_model=telescope_model,
        calibration_model=calibration_model,
        site_model=site_model_north,
        default_config=default_config,
        le_application=le_application,
        simtel_source_path=simtel_source_path,
        light_source_type=light_source_type,
        label=label,
        config_data={},
    )
    return mock_simulator_variable


@pytest.fixture
def mock_simulator_laser(
    db_config, default_config, label, model_version, simtel_path, site_model_north, io_handler
):
    simtel_source_path = simtel_path
    telescope_model = TelescopeModel(
        site="North",
        telescope_model_name="LSTN-01",
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

    # default_le_config = default_config
    le_application = "ls-beam", "layout"
    light_source_type = "laser"
    mock_simulator_laser = SimulatorLightEmission(
        telescope_model=telescope_model,
        calibration_model=calibration_model,
        site_model=site_model_north,
        default_config=default_config,
        le_application=le_application,
        simtel_source_path=simtel_source_path,
        light_source_type=light_source_type,
        label=label,
        config_data={},
    )
    return mock_simulator_laser


@pytest.fixture
def mock_output_path(label, io_handler):
    path = io_handler.get_output_directory(label)
    return path


@pytest.fixture
def calibration_model_illn(db_config, io_handler, model_version):
    calibration_model_ILLN = CalibrationModel(
        site="North",
        calibration_device_model_name="ILLN-01",
        mongo_db_config=db_config,
        model_version=model_version,
        label="test-simtel-light-emission",
    )
    return calibration_model_ILLN


def test_initialization(mock_simulator, default_config):
    assert isinstance(mock_simulator, SimulatorLightEmission)
    assert mock_simulator.le_application[0] == "xyzls"
    assert mock_simulator.light_source_type == "led"
    assert mock_simulator.default_le_config == default_config


def test_runs(mock_simulator):
    assert mock_simulator.runs == 1


def test_photons_per_run_default(mock_simulator):
    assert mock_simulator.photons_per_run == pytest.approx(1e10)


def test_from_kwargs_with_all_args(
    telescope_model_lst, calibration_model_illn, site_model_north, default_config, simtel_path
):
    kwargs = {
        "telescope_model": telescope_model_lst,
        "calibration_model": calibration_model_illn,
        "site_model": site_model_north,
        "default_le_config": default_config,
        "le_application": "xyzls",
        "label": "test_label",
        "simtel_source_path": simtel_path,
        "light_source_type": "layout",
        # "config_data": {"some_param": "value"},
    }
    simulator = SimulatorLightEmission.from_kwargs(**kwargs)

    assert isinstance(simulator, SimulatorLightEmission)
    assert simulator._telescope_model == telescope_model_lst
    assert simulator._calibration_model == calibration_model_illn
    assert simulator.le_application == "xyzls"
    assert simulator.label == "test_label"
    assert simulator._simtel_source_path == simtel_path
    assert simulator.light_source_type == "layout"


def test_from_kwargs_with_minimal_args(
    telescope_model_lst, calibration_model_illn, site_model_north, default_config, simtel_path
):
    kwargs = {
        "telescope_model": telescope_model_lst,
        "calibration_model": calibration_model_illn,
        "site_model": site_model_north,
        "default_le_config": default_config,
        "le_application": "xyzls",
        "simtel_source_path": simtel_path,
        "light_source_type": "led",
    }
    simulator = SimulatorLightEmission.from_kwargs(**kwargs)

    assert isinstance(simulator, SimulatorLightEmission)
    assert simulator._telescope_model == telescope_model_lst
    assert simulator._calibration_model == calibration_model_illn
    assert simulator.default_le_config == default_config
    assert simulator.le_application == "xyzls", "layout"
    assert simulator.label == telescope_model_lst.label
    assert simulator._simtel_source_path is not None
    assert simulator.light_source_type == "led"


@pytest.fixture
def array_layout_model(db_config, model_version, telescope_north_test_file):

    array_layout_model = ArrayLayout(
        mongo_db_config=db_config,
        model_version=model_version,
        site="North",
        telescope_list_file=telescope_north_test_file,
    )
    array_layout_model.convert_coordinates()
    return array_layout_model


def test_make_light_emission_script(
    mock_simulator,
    telescope_model_lst,
    array_layout_model,
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
        " -z 11000.0"
        " -d 0.972761,-0.10382,-0.207263"
        " -n 10000000000.0"
        " -s 300"
        " -p Gauss:0.0"
        " -a isotropic"
        f" -A {mock_output_path}/model/"
        f"{telescope_model_lst.get_parameter_value('atmospheric_profile')}"
        f" -o {mock_output_path}/xyzls.iact.gz\n"
    )

    for telescope in array_layout_model._telescope_list:  # pylint: disable=protected-access
        if telescope.name == "LSTN-01":
            xx, yy, zz = telescope.get_coordinates(crs_name="ground")

    mock_simulator.default_le_config["x_pos"]["real"] = xx
    mock_simulator.default_le_config["y_pos"]["real"] = yy
    mock_simulator.default_le_config["z_pos"]["real"] = zz

    command = mock_simulator._make_light_emission_script()

    assert command == expected_command


def test_make_light_emission_script_variable(
    mock_simulator_variable,
    telescope_model_lst,
    array_layout_model,
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
        " -n 10000000000.0"
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
    array_layout_model,
    simtel_path,
    mock_output_path,
    io_handler,
):

    for telescope in array_layout_model._telescope_list:  # pylint: disable=protected-access
        if telescope.name == "LSTN-01":
            xx, yy, zz = telescope.get_coordinates(crs_name="ground")

    mock_simulator_laser.default_le_config["x_pos"]["real"] = xx
    mock_simulator_laser.default_le_config["y_pos"]["real"] = yy
    mock_simulator_laser.default_le_config["z_pos"]["real"] = zz

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
        " --laser-position '-51627.0,5510.0,11000.0'"
        " --telescope-theta 78.037993"
        " --telescope-phi 186.091952"
        " --laser-theta 11.962007"
        " --laser-phi 173.908048"
        f" --atmosphere {mock_output_path}/model/"
        f"{mock_simulator_laser._calibration_model.get_parameter_value('atmospheric_profile')}"
        f" -o {mock_output_path}/ls-beam.iact.gz\n"
    )

    # Assert that the command matches the expected command
    assert command == expected_command


def test_calibration_pointing_direction(mock_simulator):

    # Mocking default_le_config
    mock_simulator.default_le_config = {
        "x_pos_ILLN-01": {"default": 0 * u.m},
        "y_pos_ILLN-01": {"default": 0 * u.m},
        "z_pos_ILLN-01": {"default": 0 * u.m},
        "x_pos": {"real": 100 * u.m},
        "y_pos": {"real": 200 * u.m},
        "z_pos": {"real": 400 * u.m},
    }

    # Calling calibration_pointing_direction method
    pointing_vector, angles = mock_simulator.calibration_pointing_direction()

    # Expected pointing vector and angles
    expected_pointing_vector = [0.218218, 0.436436, 0.872872]
    expected_angles = [150.794, 116.565, 150.794, -116.565]

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

    command = mock_simulator._create_postscript()

    assert command == expected_command


def test_plot_simtel_ctapipe(mock_simulator, mock_output_path):

    mock_simulator.output_directory = "./tests/resources/"
    fig = mock_simulator.plot_simtel_ctapipe(return_cleaned=0)
    assert isinstance(fig, plt.Figure)  # Check if fig is an instance of matplotlib figure


def test_make_simtel_script(mock_simulator):
    # Create a mock file content
    mock_file_content = "Sample content of config file"

    # Patch the built-in open function to return a mock file
    with patch("builtins.open", mock_open(read_data=mock_file_content)) as mock_file:
        # Reset the mock's call count to zero
        mock_file.reset_mock()

        # Mock the necessary attributes and methods used within _make_simtel_script
        mock_simulator._simtel_source_path = MagicMock()
        mock_simulator._telescope_model = MagicMock()
        mock_simulator._site_model = MagicMock()

        mock_simulator._simtel_source_path.joinpath.return_value = (
            "/path/to/sim_telarray/bin/sim_telarray/"
        )
        mock_simulator._telescope_model.get_config_file.return_value = "/path/to/config.cfg"
        mock_simulator._telescope_model.get_parameter_value.side_effect = lambda param: (
            "atm_test" if param == "atmospheric_transmission" else MagicMock()
        )
        mock_simulator._site_model.get_parameter_value.side_effect = lambda param: (
            "999" if param == "corsika_observation_level" else MagicMock()
        )

        mock_simulator.output_directory = "/directory"

        expected_command = (
            "/path/to/sim_telarray/bin/sim_telarray/ -c /path/to/config.cfg "
            "-DNUM_TELESCOPES=1 -I../cfg/CTAiobuf_maximum=1000000000 "
            "-C altitude=999 -C atmospheric_transmission=atm_test "
            "-C TRIGGER_CURRENT_LIMIT=20 -C TRIGGER_TELESCOPES=1 "
            "-C TELTRIG_MIN_SIGSUM=7.8 -C PULSE_ANALYSIS=-30 "
            "-C telescope_theta=0 -C telescope_phi=0 "
            "-C power_law=2.68 -C input_file=/directory/xyzls.iact.gz "
            "-C output_file=/directory/xyzls_layout.simtel.gz\n"
        )

        # Call the method under test
        command = mock_simulator._make_simtel_script()

        mock_file.assert_has_calls(
            [
                call("/path/to/config.cfg", "r", encoding="utf-8"),
                call().__enter__(),
                call().readlines(),
                call().__exit__(None, None, None),
                call("/path/to/config.cfg", "w", encoding="utf-8"),
                call().__enter__(),
                call().write("Sample content of config file"),
                call().__exit__(None, None, None),
            ],
            any_order=False,
        )

        assert command == expected_command


@patch("os.system")
@patch("simtools.simtel.simtel_light_emission.SimulatorLightEmission._make_light_emission_script")
@patch("simtools.simtel.simtel_light_emission.SimulatorLightEmission._make_simtel_script")
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
    script_path = mock_simulator.prepare_script(generate_postscript=True)

    # Assertions
    mock_os_system.assert_called_once_with(f"chmod ug+x {script_path}")
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
    print(mock_simulator._script_dir)
    # Ensure correct return value
    assert str(script_path) == f"{mock_simulator._script_dir}/xyzls-lightemission.sh"

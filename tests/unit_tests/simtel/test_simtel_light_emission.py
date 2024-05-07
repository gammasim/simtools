from unittest.mock import MagicMock, Mock, patch

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
        telescope_model,
        calibration_model,
        site_model_north,
        default_config,
        le_application,
        simtel_source_path,
        light_source_type,
        label,
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
        telescope_model,
        calibration_model,
        site_model_north,
        default_config,
        le_application,
        simtel_source_path,
        light_source_type,
        label,
        config_data={},
    )
    return mock_simulator_variable


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
    assert mock_simulator.photons_per_run == 1e10


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
    # assert simulator.config == {"beam_shape": "Gauss"}


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
        " -d 0.9727607415193217,-0.10381993309259616,-0.20726302432278723"
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
    expected_angles = [29.205932, 63.434949, 150.794068, -116.565051]

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
    # Mock necessary objects and functions
    with (
        patch("os.path.exists", return_value=True),
        patch("ctapipe.io.EventSource") as mock_event_source,
        patch("ctapipe.visualization.CameraDisplay") as mock_camera_display,
        patch(
            "matplotlib.pyplot.subplots", return_value=(plt.figure(), plt.axes())
        ) as mock_subplots,
    ):
        mock_event_source = MagicMock()
        mock_event_source_instance = mock_event_source.return_value
        mock_event_source_instance.subarray.tel[1].camera.geometry = MagicMock()
        mock_event_source_instance.r1.tel.keys.return_value = [1]  # Mocking tel keys

        mock_event = MagicMock()
        mock_event_instance = mock_event.return_value
        mock_event_instance.dl1.tel.return_value.image = MagicMock()

        mock_camera_display = MagicMock()
        mock_camera_display_instance = mock_camera_display.return_value

        # Configure the mock behavior of EventSource
        mock_event_source_instance.return_value.__enter__.return_value.__iter__.return_value = [
            {
                "index": {"event_id": 1, "obs_id": 1},
                "r1": {"tel": {1: {}}},
                "dl1": {"tel": {1: {"image": MagicMock()}}},
            }
        ]
        # supply file from resources
        mock_simulator.output_directory = "tests/resources/"
        fig = mock_simulator.plot_simtel_ctapipe(return_cleaned=0)

    assert isinstance(fig, plt.Figure)  # Check if fig is an instance of matplotlib figure
    assert mock_event_source.called_once_with(
        f"{mock_output_path}/xyzls_layout.simtel.gz", max_events=1
    )
    assert mock_event.called_once()  # Check if event is called
    assert mock_event_instance.dl1.tel.called_once_with(
        1
    )  # Check if dl1.tel is called with the correct argument
    assert mock_camera_display.called_once_with(
        mock_event_source_instance.subarray.tel[1].camera.geometry,
        image=mock_event_instance.dl1.tel().image,
        norm="symlog",
        ax=mock_camera_display_instance,
    )
    assert (
        mock_camera_display_instance.add_colorbar.called_once()
    )  # Check if add_colorbar is called
    assert mock_camera_display_instance.set_limits_percent.called_once_with(
        100
    )  # Check if set_limits_percent is called
    assert mock_subplots.called_once_with(
        1, 1, dpi=300
    )  # Check if subplots is called with correct arguments


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

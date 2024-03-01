from unittest.mock import MagicMock, Mock, patch

import astropy.units as u
import matplotlib.pyplot as plt
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
            "default": 100000 * u.cm,
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
def mock_simulator(db_config, default_config, label, simtel_path, io_handler):
    version = "Released"
    simtel_source_path = simtel_path
    tel = TelescopeModel(
        site="North",
        telescope_model_name="LST-1",
        model_version=version,
        label="test-simtel-light-emission",
        mongo_db_config=db_config,
    )
    # default_le_config = default_config
    le_application = "xyzls"
    mock_simulator = SimulatorLightEmission(
        tel, default_config, le_application, simtel_source_path, label, config_data={}
    )
    return mock_simulator


@pytest.fixture
def mock_output_path(label, io_handler):
    path = io_handler.get_output_directory(label)
    return path


def test_initialization(mock_simulator, default_config):
    assert isinstance(mock_simulator, SimulatorLightEmission)
    assert mock_simulator.le_application == "xyzls"
    assert (
        mock_simulator.default_le_config == default_config
    )  # Update with the expected default configuration


def test_runs(mock_simulator):
    assert mock_simulator.runs == 1


def test_photons_per_run_default(mock_simulator):
    assert mock_simulator.photons_per_run == 1e10


def test_from_kwargs_with_all_args(telescope_model_lst, default_config, simtel_path):
    kwargs = {
        "telescope_model": telescope_model_lst,
        "default_le_config": default_config,
        "le_application": "xyzls",
        "label": "test_label",
        "simtel_source_path": simtel_path,
        # "config_data": {"some_param": "value"},
    }
    simulator = SimulatorLightEmission.from_kwargs(**kwargs)

    assert isinstance(simulator, SimulatorLightEmission)
    assert simulator._telescope_model == telescope_model_lst
    # assert simulator.default_le_config == default_config
    assert simulator.le_application == "xyzls"
    assert simulator.label == "test_label"
    assert simulator._simtel_source_path == simtel_path
    # assert simulator.config == {"beam_shape": "Gauss"}


def test_from_kwargs_with_minimal_args(telescope_model_lst, default_config, simtel_path):
    kwargs = {
        "telescope_model": telescope_model_lst,
        "default_le_config": default_config,
        "le_application": "xyzls",
        "simtel_source_path": simtel_path,
    }
    simulator = SimulatorLightEmission.from_kwargs(**kwargs)

    assert isinstance(simulator, SimulatorLightEmission)
    assert simulator._telescope_model == telescope_model_lst
    assert simulator.default_le_config == default_config
    assert simulator.le_application == "xyzls"
    assert simulator.label == telescope_model_lst.label
    assert simulator._simtel_source_path is not None
    # assert simulator.config == simulator.light_emission_default_configuration()


def test_make_light_emission_script(
    mock_simulator, telescope_model_lst, simtel_path, mock_output_path, io_handler
):
    expected_command = (
        f" rm {mock_output_path}/xyzls.simtel.gz\n"
        f"{simtel_path}/sim_telarray/LightEmission/xyzls"
        " -n 10000000000.0"
        " -x 0.0"
        " -y 0.0"
        " -z 100000.0"
        " -d 0,0.0,-1"
        f" -A {mock_output_path}/model/"
        f"{telescope_model_lst.get_parameter_value('atmospheric_profile')}"
        f" -o {mock_output_path}/xyzls.iact.gz\n"
    )

    command = mock_simulator._make_light_emission_script()

    assert command == expected_command


def test_create_postscript(mock_simulator, simtel_path, mock_output_path):
    expected_command = (
        f"{simtel_path}/hessioxxx/bin/read_cta"
        " --min-tel 1 --min-trg-tel 1"
        " -q --integration-scheme 4 --integration-window 7,3 -r 5"
        " --plot-with-sum-only"
        " --plot-with-pixel-amp --plot-with-pixel-id"
        f" -p {mock_output_path}/xyzls.ps"
        f" {mock_output_path}/xyzls.simtel.gz\n"
    )

    command = mock_simulator._create_postscript()

    assert command == expected_command


def test_plot_simtel_ctapipe(mock_simulator):
    # Mock necessary objects and functions
    mock_event_source = MagicMock()
    mock_event_source_instance = mock_event_source.return_value
    mock_event_source_instance.subarray.tel[1].camera.geometry = MagicMock()
    mock_event_source_instance.r1.tel.keys.return_value = [1]  # Mocking tel keys

    mock_event = MagicMock()
    mock_event_instance = mock_event.return_value
    mock_event_instance.dl1.tel.return_value.image = MagicMock()

    mock_camera_display = MagicMock()
    mock_camera_display_instance = mock_camera_display.return_value

    with patch("os.path.exists", return_value=True), patch(
        "ctapipe.io.EventSource", mock_event_source
    ), patch("ctapipe.calib.CameraCalibrator"), patch(
        "ctapipe.visualization.CameraDisplay", mock_camera_display
    ), patch(
        "matplotlib.pyplot.subplots", return_value=(plt.figure(), MagicMock())
    ):
        # Configure the mock behavior of EventSource
        mock_event_source_instance.return_value.__enter__.return_value.__iter__.return_value = [
            {
                "index": {"event_id": 1, "obs_id": 1},
                "r1": {"tel": {1: {}}},
                "dl1": {"tel": {1: {"image": MagicMock()}}},
            }
        ]
        # TODO fix the patch
        fig = mock_simulator.plot_simtel_ctapipe()

    assert isinstance(fig, plt.Figure)  # Check if fig is an instance of matplotlib figure
    assert mock_event_source.called_once_with(f"{mock_output_path}/xyzls.simtel.gz", max_events=1)
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
    assert plt.subplots.called_once_with(
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
    assert mock_os_system.called_with(f"chmod ug+x {script_path}")
    assert mock_make_light_emission_script.called
    assert mock_make_simtel_script.called
    assert mock_open.called_with(f"{mock_simulator._script_file}", "w", encoding="utf-8")
    assert mock_file.write.called_with("#!/usr/bin/env bash\n\n")
    assert mock_file.write.called_with("light_emission_script_command\n\n")
    assert mock_file.write.called_with("simtel_script_command\n\n")
    assert mock_file.write.called_with("# Generate postscript\n\n")
    assert mock_file.write.called_with(
        "postscript_command\n\n"
    )  # Assuming this is generated by _create_postscript method
    assert mock_file.write.called_with("# End\n\n")
    assert script_path == mock_simulator._script_file

#!/usr/bin/python3

import logging
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock, call, mock_open, patch

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

import simtools.simtel.simulator_light_emission as sim_mod
from simtools.model.calibration_model import CalibrationModel
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.simulator_light_emission import SimulatorLightEmission
from simtools.utils import general as gen
from simtools.visualization.light_emission_plots import plot_simtel_ctapipe

SIM_MOD_PATH = "simtools.simtel.simulator_light_emission"


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
        light_emission_config={},
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
        light_emission_config=default_config,
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
        light_emission_config=default_config,
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
    assert mock_simulator.light_emission_config == {}


def test_initialization_variable(mock_simulator_variable, default_config):
    assert isinstance(mock_simulator_variable, SimulatorLightEmission)
    assert mock_simulator_variable.le_application[0] == "xyzls"
    assert mock_simulator_variable.light_source_type == "led"
    assert mock_simulator_variable.light_emission_config == default_config


def test_runs(mock_simulator):
    assert mock_simulator.runs == 1


def test_photons_per_run_default(mock_simulator):
    assert mock_simulator.photons_per_run == pytest.approx(1e8)


def test_make_light_emission_script(
    mock_simulator,
    site_model_north,
    mock_output_path,
):
    """layout coordinate vector between LST and ILLN"""
    expected_command = (
        f"rm {mock_output_path}/xyzls_layout.simtel.gz\n"
        f"sim_telarray/LightEmission/xyzls"
        f" -h  {site_model_north.get_parameter_value('corsika_observation_level')}"
        f" --telpos-file {mock_output_path}/telpos.dat"
        " -x -58717.99999999999"
        " -y 275.0"
        " -z 13700.0"
        " -d 0.979101,-0.104497,-0.174477"
        " -n 100000000.0"
        " -s 300"
        " -p Gauss:0.0"
        " -a isotropic"
        f" -A {mock_output_path}/model/6.0.0/"
        f"{site_model_north.get_parameter_value('atmospheric_profile')}"
        f" -o {mock_output_path}/xyzls.iact.gz\n"
    )

    command = mock_simulator._make_light_emission_script()

    assert command == expected_command


def test_make_light_emission_script_variable(
    mock_simulator_variable,
    site_model_north,
    mock_output_path,
):
    """layout coordinate vector between LST and ILLN"""
    expected_command = (
        f"rm {mock_output_path}/xyzls_variable.simtel.gz\n"
        f"sim_telarray/LightEmission/xyzls"
        f" -h  {site_model_north.get_parameter_value('corsika_observation_level')}"
        f" --telpos-file {mock_output_path}/telpos.dat"
        " -x 0.0"
        " -y 0.0"
        " -z 100000.0"
        " -d 0,0,-1"
        " -n 100000000.0"
        f" -A {mock_output_path}/model/6.0.0/"
        f"{site_model_north.get_parameter_value('atmospheric_profile')}"
        f" -o {mock_output_path}/xyzls.iact.gz\n"
    )
    assert mock_simulator_variable.le_application[0] == "xyzls"
    assert mock_simulator_variable.le_application[1] == "variable"

    command = mock_simulator_variable._make_light_emission_script()

    assert command == expected_command


def test_make_light_emission_script_laser(
    mock_simulator_laser,
    site_model_north,
    mock_output_path,
):
    command = mock_simulator_laser._make_light_emission_script()

    expected_command = (
        f"rm {mock_output_path}/ls-beam_layout.simtel.gz\n"
        f"sim_telarray/LightEmission/ls-beam"
        f" -h  {site_model_north.get_parameter_value('corsika_observation_level')}"
        f" --telpos-file {mock_output_path}/telpos.dat"
        " --events 1"
        " --bunches 2500000"
        " --step 0.1"
        " --bunchsize 1"
        " --spectrum 300"
        " --lightpulse Gauss:0.0"
        " --laser-position '-51627.0,5510.0,9200.0'"
        " --telescope-theta 79.951773"
        " --telescope-phi 186.091952"
        " --laser-theta 10.048226999999997"
        " --laser-phi 173.908048"
        f" --atmosphere {mock_output_path}/model/6.0.0/"
        f"{site_model_north.get_parameter_value('atmospheric_profile')}"
        f" -o {mock_output_path}/ls-beam.iact.gz\n"
    )

    assert command == expected_command


def test_calibration_pointing_direction(mock_simulator):
    pointing_vector, angles = mock_simulator.calibration_pointing_direction()

    expected_pointing_vector = [0.979, -0.104, -0.174]
    expected_angles = [79.952, 186.092, 79.952, 173.908]

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
    mock_simulator.distance = 1000 * u.m

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
        mock_simulator._calibration_model = MagicMock()

        mock_simulator.calibration_pointing_direction = MagicMock(
            return_value=([0, 0, 1], [76.980826, 180.17047, 0, 0])
        )

        mock_simulator._simtel_path.joinpath.return_value = (
            "/path/to/sim_telarray/bin/sim_telarray/"
        )
        path_to_config_directory = "/path/to/config/"
        mock_simulator._telescope_model.config_file_directory = path_to_config_directory
        path_to_config_file = "/path/to/config/config.cfg"
        config_file_path_mock = PropertyMock(return_value=path_to_config_file)
        type(mock_simulator._telescope_model).config_file_path = config_file_path_mock

        # Patch Path.open to mock file handling for the config file
        with patch("pathlib.Path.open", mock_open(read_data=mock_file_content)) as mock_path_open:

            def get_telescope_model_param(param):
                if param == "atmospheric_transmission":
                    return "atm_test"
                if param == "array_element_position_ground":
                    return (1, 1, 1)
                return MagicMock()

            mock_simulator._telescope_model.get_parameter_value.side_effect = (
                get_telescope_model_param
            )

            mock_simulator._site_model.get_parameter_value_with_unit.return_value = 999 * u.m
            mock_simulator._site_model.get_parameter_value.side_effect = lambda param: (
                "999" if param == "corsika_observation_level" else MagicMock()
            )

            mock_simulator.output_directory = "/directory"

            expected_command = (
                "SIM_TELARRAY_CONFIG_PATH='' "
                "/path/to/sim_telarray/bin/sim_telarray/ "
                "-I/path/to/config/ -I/path/to/sim_telarray/bin/sim_telarray/ "
                "-c /path/to/config/config.cfg "
                "-DNUM_TELESCOPES=1 "
                "-C altitude=999.0 -C atmospheric_transmission=atm_test "
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

            mock_path_open.assert_has_calls(
                [
                    call("r", encoding="utf-8"),
                    call("w", encoding="utf-8"),
                ],
                any_order=True,
            )


@patch("os.system")
@patch(f"{SIM_MOD_PATH}.SimulatorLightEmission._make_light_emission_script")
@patch(f"{SIM_MOD_PATH}.SimulatorLightEmission._make_simtel_script")
@patch("builtins.open", create=True)
def test_prepare_script(
    mock_open,
    mock_make_simtel_script,
    mock_make_light_emission_script,
    mock_os_system,
    mock_simulator,
):
    mock_file = Mock()
    mock_open.return_value.__enter__.return_value = mock_file
    mock_make_light_emission_script.return_value = "light_emission_script_command"
    mock_make_simtel_script.return_value = "simtel_script_command"
    mock_simulator.distance = 1000 * u.m
    script_path = mock_simulator.prepare_script(
        generate_postscript=True, integration_window=["7", "3"], level="5"
    )

    assert gen.program_is_executable(script_path)
    mock_make_light_emission_script.assert_called_once()  # Ensure this mock is called
    mock_make_simtel_script.assert_called_once()  # Ensure this mock is called

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


def test_update_light_emission_config(mock_simulator_variable):
    """Test updating the light emission configuration."""
    # Valid key
    mock_simulator_variable.update_light_emission_config("z_pos", [200 * u.cm, 300 * u.cm])
    assert mock_simulator_variable.light_emission_config["z_pos"]["default"] == [
        200 * u.cm,
        300 * u.cm,
    ]

    # Invalid key
    with pytest.raises(
        KeyError, match="Key 'invalid_key' not found in light emission configuration."
    ):
        mock_simulator_variable.update_light_emission_config("invalid_key", 100 * u.cm)


def test_distance_list(mock_simulator_variable):
    """Test converting a list of distances to astropy.Quantity."""
    distances = mock_simulator_variable.distance_list(["100", "200", "300"])
    assert distances == [100 * u.m, 200 * u.m, 300 * u.m]

    # Invalid input
    with pytest.raises(ValueError, match="Distances must be numeric values"):
        mock_simulator_variable.distance_list(["100", "invalid", "300"])


def test_calculate_distance_telescope_calibration_device_layout(mock_simulator):
    """Test distance calculation for layout positions."""
    distances = mock_simulator.calculate_distance_telescope_calibration_device()
    assert len(distances) == 1
    assert isinstance(distances[0], u.Quantity)


def test_calculate_distance_telescope_calibration_device_variable(mock_simulator_variable):
    """Test distance calculation for variable positions."""
    mock_simulator_variable.light_emission_config["z_pos"]["default"] = [100 * u.m, 200 * u.m]
    distances = mock_simulator_variable.calculate_distance_telescope_calibration_device()
    assert len(distances) == 2
    assert distances[0].unit == u.m
    assert distances[1].unit == u.m
    assert distances[0].value == pytest.approx(100)
    assert distances[1].value == pytest.approx(200)


@patch(f"{SIM_MOD_PATH}.SimulatorLightEmission.run_simulation")
@patch(f"{SIM_MOD_PATH}.SimulatorLightEmission.save_figures_to_pdf")
def test_simulate_variable_distances(
    mock_save_figures, mock_run_simulation, mock_simulator_variable
):
    """Test simulating light emission for variable distances."""
    mock_simulator_variable.light_emission_config["z_pos"]["default"] = [100 * u.m, 200 * u.m]
    args_dict = {"distances_ls": None, "telescope": "LSTN-01"}

    mock_simulator_variable.simulate_variable_distances(args_dict)

    assert mock_run_simulation.call_count == 2

    mock_save_figures.assert_called_once()


@patch(f"{SIM_MOD_PATH}.SimulatorLightEmission.run_simulation")
@patch(f"{SIM_MOD_PATH}.SimulatorLightEmission.save_figures_to_pdf")
def test_simulate_layout_positions(mock_save_figures, mock_run_simulation, mock_simulator):
    """Test simulating light emission for layout positions."""
    args_dict = {"telescope": "LSTN-01"}

    mock_simulator.simulate_layout_positions(args_dict)

    mock_run_simulation.assert_called_once()

    mock_save_figures.assert_called_once()


@patch(f"{SIM_MOD_PATH}.SimulatorLightEmission._plot_simulation_output")
@patch(f"{SIM_MOD_PATH}.SimulatorLightEmission._get_distance_for_plotting")
@patch(f"{SIM_MOD_PATH}.SimulatorLightEmission._get_simulation_output_filename")
def test_process_simulation_output(
    mock_get_simulation_output_filename,
    mock_get_distance_for_plotting,
    mock_plot_simulation_output,
    mock_simulator_variable,
):
    """Test the process_simulation_output method."""
    args_dict = {
        "boundary_thresh": 5,
        "picture_thresh": 3,
        "min_neighbors": 2,
        "return_cleaned": True,
    }
    figures = []

    # Mock return values
    mock_get_simulation_output_filename.return_value = "mock_filename.simtel.gz"
    mock_get_distance_for_plotting.return_value = 1000 * u.m
    mock_figure = Mock()
    mock_plot_simulation_output.return_value = mock_figure

    # Call the method
    mock_simulator_variable.process_simulation_output(args_dict, figures)

    # Assertions
    mock_get_simulation_output_filename.assert_called_once()
    mock_get_distance_for_plotting.assert_called_once()
    mock_plot_simulation_output.assert_called_once_with(
        "mock_filename.simtel.gz",
        5,
        3,
        2,
        1000 * u.m,
        True,
    )
    assert len(figures) == 1
    assert figures[0] == mock_figure


def test_get_simulation_output_filename(mock_simulator_variable):
    """Test the _get_simulation_output_filename method."""
    mock_simulator_variable.output_directory = "./tests/resources/"
    mock_simulator_variable.le_application = ("xyzls", "variable")

    filename = mock_simulator_variable._get_simulation_output_filename()

    expected_filename = "./tests/resources//xyzls_variable.simtel.gz"
    assert filename == expected_filename


def test_get_distance_for_plotting_with_z_pos(mock_simulator_variable):
    """Test _get_distance_for_plotting when z_pos is available."""
    mock_simulator_variable.light_emission_config = {"z_pos": {"default": 1000 * u.m}}

    distance = mock_simulator_variable._get_distance_for_plotting()

    assert distance == 1000 * u.m


@patch(f"{SIM_MOD_PATH}.SimulatorLightEmission._get_simulation_output_filename")
def test_process_simulation_output_attribute_error(
    mock_get_simulation_output_filename, mock_simulator_variable
):
    """Test process_simulation_output handles AttributeError and logs a warning."""
    args_dict = {
        "boundary_thresh": 5,
        "picture_thresh": 3,
        "min_neighbors": 2,
        "return_cleaned": True,
    }
    figures = []

    # Simulate AttributeError
    mock_get_simulation_output_filename.side_effect = AttributeError

    with patch.object(mock_simulator_variable._logger, "warning") as mock_warning:
        mock_simulator_variable.process_simulation_output(args_dict, figures)

        # Assert the warning was logged
        mock_warning.assert_called_once_with(
            "Telescope not triggered at distance of "
            f"{mock_simulator_variable.light_emission_config['z_pos']['default']}"
        )


def test_get_distance_for_plotting(mock_simulator_variable):
    """Test the _get_distance_for_plotting method."""
    mock_simulator_variable.light_emission_config = {"z_pos": {"default": 1000 * u.m}}
    distance = mock_simulator_variable._get_distance_for_plotting()
    assert distance == 1000 * u.m

    mock_simulator_variable.light_emission_config = {}
    mock_simulator_variable.distance = 1500.0001 * u.m

    with patch.object(mock_simulator_variable._logger, "warning"):
        distance = mock_simulator_variable._get_distance_for_plotting()
        # allow rounding to nearest meter
        assert np.isclose(distance.to_value(u.m), 1500.0)


@patch(f"{SIM_MOD_PATH}.save_figs_to_pdf")
def test_save_figures_to_pdf(mock_save_figs_to_pdf, mock_simulator_variable):
    """Test the save_figures_to_pdf method."""
    figures = [Mock(), Mock()]  # Mock figures
    telescope = "LSTN-01"

    mock_simulator_variable.save_figures_to_pdf(figures, telescope)

    # Assert save_figs_to_pdf was called with the correct arguments
    mock_save_figs_to_pdf.assert_called_once_with(
        figures,
        f"{mock_simulator_variable.output_directory}/"
        f"{telescope}_{mock_simulator_variable.le_application[0]}_{mock_simulator_variable.le_application[1]}.pdf",
    )


@patch(f"{SIM_MOD_PATH}.SimulatorLightEmission.process_simulation_output")
@patch("subprocess.run")
@patch("builtins.open", new_callable=mock_open)
def test_run_simulation(
    mock_open, mock_subprocess_run, mock_process_simulation_output, mock_simulator_variable
):
    """Test the run_simulation method."""
    args_dict = {
        "boundary_thresh": 5,
        "picture_thresh": 3,
        "min_neighbors": 2,
        "return_cleaned": True,
    }
    figures = []

    mock_simulator_variable.prepare_script = Mock(return_value="dummy_script.sh")
    mock_simulator_variable.run_simulation(args_dict, figures)

    # first open for write, then append via FileHandler
    assert mock_open.call_count >= 1
    first_call = mock_open.call_args_list[0]
    assert first_call.args[0] == Path(mock_simulator_variable.output_directory) / "logfile.log"

    mock_subprocess_run.assert_called_once_with(
        "dummy_script.sh",
        shell=False,
        check=False,
        text=True,
        stdout=mock_open.return_value.__enter__.return_value,
        stderr=mock_open.return_value.__enter__.return_value,
    )

    # Assert process_simulation_output was called
    mock_process_simulation_output.assert_called_once_with(args_dict, figures)


def test_write_telpos_file(mock_simulator, tmp_path):
    """
    Test the _write_telpos_file method to ensure it writes the correct telescope positions.
    """
    # Mock the output directory to use a temporary path
    mock_simulator.output_directory = tmp_path

    # Mock the telescope model
    mock_simulator._telescope_model = MagicMock()

    mock_simulator._telescope_model.get_parameter_value_with_unit.side_effect = (
        lambda param, *args: {
            "array_element_position_ground": (1.0 * u.m, 2.0 * u.m, 3.0 * u.m),
            "telescope_sphere_radius": 4.0 * u.m,
        }[param]
    )

    # Call the method to write the telpos file
    telpos_file = mock_simulator._write_telpos_file()

    # Verify the file was created and has the correct content
    assert telpos_file.exists()

    # Read the content of the file
    with open(telpos_file) as f:
        content = f.read().strip()

    # Check that the content contains the expected values converted to cm
    # 1m = 100cm, 2m = 200cm, 3m = 300cm, 4m = 400cm
    assert content == "100.0 200.0 300.0 400.0"


def _make_dummy_fig():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    return fig


@pytest.fixture
def sim_instance():
    # Create instance without running __init__ to avoid heavy deps
    inst = object.__new__(SimulatorLightEmission)
    inst._logger = logging.getLogger(SIM_MOD_PATH)

    def fake_calib(filename, args_dict, distance, figures):
        figures.append(_make_dummy_fig())

    inst._plot_calibration_outputs = fake_calib
    return inst


def test_plot_flasher_outputs_success(monkeypatch, sim_instance, caplog):
    # Stub visualize functions to return figures
    monkeypatch.setattr(sim_mod, "plot_simtel_time_traces", lambda *a, **k: _make_dummy_fig())
    monkeypatch.setattr(sim_mod, "plot_simtel_peak_timing", lambda *a, **k: _make_dummy_fig())
    monkeypatch.setattr(
        sim_mod, "plot_simtel_waveform_pcolormesh", lambda *a, **k: _make_dummy_fig()
    )
    # New integrated images
    monkeypatch.setattr(
        sim_mod, "plot_simtel_integrated_signal_image", lambda *a, **k: _make_dummy_fig()
    )
    monkeypatch.setattr(
        sim_mod, "plot_simtel_integrated_pedestal_image", lambda *a, **k: _make_dummy_fig()
    )

    figures = []
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=SIM_MOD_PATH):
        sim_instance._plot_flasher_outputs("dummy.simtel.gz", {"n_trace_pixels": 6}, None, figures)

    # 1 calibration + 5 plots (signal, pedestal, traces, peak timing, pcolormesh)
    assert len(figures) == 6

    messages = "\n".join(r.message for r in caplog.records)
    assert "Added integrated signal image" in messages
    assert "Added integrated pedestal image" in messages
    assert "Added time-trace figure" in messages
    assert "Added peak timing figure" in messages
    assert "Added waveform pcolormesh figure" in messages

    # Close figures
    for f in figures:
        plt.close(f)


def test_plot_flasher_outputs_handles_errors(monkeypatch, sim_instance, caplog):
    # Stub visualize functions to raise exceptions
    def _raise(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(sim_mod, "plot_simtel_time_traces", _raise)
    monkeypatch.setattr(sim_mod, "plot_simtel_peak_timing", _raise)
    monkeypatch.setattr(sim_mod, "plot_simtel_waveform_pcolormesh", _raise)
    # Integrated images return None to simulate absence without raising
    monkeypatch.setattr(sim_mod, "plot_simtel_integrated_signal_image", lambda *a, **k: None)
    monkeypatch.setattr(sim_mod, "plot_simtel_integrated_pedestal_image", lambda *a, **k: None)

    figures = []
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=SIM_MOD_PATH):
        sim_instance._plot_flasher_outputs("dummy.simtel.gz", {"n_trace_pixels": 3}, None, figures)

    # Only calibration figure appended
    assert len(figures) == 1

    messages = "\n".join(r.message for r in caplog.records)
    assert "No event time traces available" in messages or "No event time" in messages
    assert "Peak timing plot not available" in messages
    assert "Waveform pcolormesh not available" in messages

    for f in figures:
        plt.close(f)


def test_add_flasher_options():
    inst = object.__new__(SimulatorLightEmission)
    inst._flasher_model = MagicMock()
    inst._telescope_model = MagicMock()
    inst.runs = 1
    inst.photons_per_run = 1.23e6

    def gpvu(name):
        mp = {
            "flasher_position": [0.5 * u.cm, -1.5 * u.cm],
            "flasher_depth": 250.0 * u.cm,
            "spectrum": 405 * u.nm,
        }
        return mp[name]

    inst._flasher_model.get_parameter_value_with_unit.side_effect = gpvu
    inst._flasher_model.get_parameter_value.side_effect = lambda n: {
        "lightpulse": "Gauss:3.2",
        "angular_distribution": "isotropic",
        "bunch_size": 2.0,
    }[n]

    inst._telescope_model.get_parameter_value_with_unit.return_value = 120.0 * u.cm

    cmd = inst._add_flasher_options("")

    assert "--events 1" in cmd
    assert "--photons 1230000.0" in cmd
    assert "--bunchsize 2.0" in cmd
    assert "--xy 0.5,-1.5" in cmd
    assert "--distance 250.0" in cmd
    assert "--camera-radius 60.0" in cmd
    assert "--spectrum 405" in cmd
    assert "--lightpulse Gauss:3.2" in cmd
    assert "--angular-distribution isotropic" in cmd


def test_add_flasher_command_options_branch():
    inst = object.__new__(SimulatorLightEmission)
    inst._telescope_model = MagicMock()
    inst._flasher_model = MagicMock()

    with patch.object(inst, "_add_flasher_options", return_value="mst") as mst:
        inst._telescope_model.name = "SSTS-05"
        out = inst._add_flasher_command_options("")
        assert out == "mst"
        mst.assert_called_once()

    with patch.object(inst, "_add_flasher_options", return_value="mst") as mst:
        inst._telescope_model.name = "MSTN-04"
        out = inst._add_flasher_command_options("")
        assert out == "mst"
        mst.assert_called_once()


def test_get_distance_for_plotting_flasher():
    inst = object.__new__(SimulatorLightEmission)
    inst.light_source_type = "flasher"
    inst._flasher_model = MagicMock()
    inst._flasher_model.get_parameter_value_with_unit.return_value = 150.0 * u.cm

    d = inst._get_distance_for_plotting()
    assert d.to_value(u.m) == pytest.approx(1.5)


def test_prepare_ff_atmosphere_files_creates_aliases(tmp_path, caplog):
    inst = object.__new__(SimulatorLightEmission)
    inst._logger = logging.getLogger(SIM_MOD_PATH)
    inst._telescope_model = MagicMock()
    inst._telescope_model.get_parameter_value.return_value = "atm_test_profile.dat"

    src = tmp_path / "atm_test_profile.dat"
    src.write_text("atmcontent", encoding="utf-8")

    with caplog.at_level(logging.DEBUG, logger=SIM_MOD_PATH):
        rid = inst._prepare_ff_atmosphere_files(tmp_path)

    assert rid == 1
    for name in ("atmprof1.dat", "atm_profile_model_1.dat"):
        alias = tmp_path / name
        assert alias.exists()
        # Content must match source (works for symlink or copied file)
        assert alias.read_text(encoding="utf-8") == "atmcontent"


def test_prepare_ff_atmosphere_files_copy_fallback(tmp_path, monkeypatch):
    inst = object.__new__(SimulatorLightEmission)
    inst._logger = logging.getLogger(SIM_MOD_PATH)
    inst._telescope_model = MagicMock()
    inst._telescope_model.get_parameter_value.return_value = "atm_prof2.dat"

    src = tmp_path / "atm_prof2.dat"
    src.write_text("X", encoding="utf-8")

    def raise_oserr(self, target):
        raise OSError("symlink not permitted")

    monkeypatch.setattr(Path, "symlink_to", raise_oserr, raising=True)

    def fake_copy2(src_path, dst_path):
        Path(dst_path).write_bytes(Path(src_path).read_bytes())
        return str(dst_path)

    monkeypatch.setattr(shutil, "copy2", fake_copy2, raising=True)

    rid = inst._prepare_ff_atmosphere_files(tmp_path)
    assert rid == 1

    a1 = tmp_path / "atmprof1.dat"
    a2 = tmp_path / "atm_profile_model_1.dat"
    assert a1.exists()
    assert a2.exists()
    assert not a1.is_symlink()
    assert not a2.is_symlink()
    assert a1.read_text(encoding="utf-8") == "X"
    assert a2.read_text(encoding="utf-8") == "X"


def test_build_altitude_atmo_block_ff1m(tmp_path, monkeypatch):
    inst = object.__new__(SimulatorLightEmission)
    inst._logger = logging.getLogger(SIM_MOD_PATH)
    inst._simtel_path = Path("/simroot")

    monkeypatch.setattr(inst, "_prepare_ff_atmosphere_files", lambda *_: 42)

    block = inst._build_altitude_atmo_block("ff-1m", tmp_path, 2150 * u.m, tmp_path / "telpos.dat")

    assert " -I." in block
    assert f" -I{inst._simtel_path.joinpath('sim_telarray/cfg')}" in block
    assert f" -I{tmp_path}" in block
    assert "--altitude 2150.0" in block
    assert "--atmosphere 42" in block


def test_build_altitude_atmo_block_default(tmp_path):
    inst = object.__new__(SimulatorLightEmission)
    inst._logger = logging.getLogger(SIM_MOD_PATH)
    inst._simtel_path = Path("/simroot")

    telpos = tmp_path / "telpos.dat"
    block = inst._build_altitude_atmo_block("xyzls", tmp_path, 2100 * u.m, telpos)
    assert block == f" -h  2100.0 --telpos-file {telpos}"


def test_build_source_specific_block_branches(tmp_path, caplog, monkeypatch):
    inst = object.__new__(SimulatorLightEmission)
    inst._logger = logging.getLogger(SIM_MOD_PATH)

    monkeypatch.setattr(inst, "_add_flasher_command_options", lambda cmd: "FLASHER")
    monkeypatch.setattr(inst, "_add_led_command_options", lambda cmd: "LED")
    monkeypatch.setattr(
        inst,
        "_add_laser_command_options",
        lambda cmd, x, y, z, cfg: "LASER" if cfg == tmp_path else "BAD",
    )

    inst.light_source_type = None

    out = inst._build_source_specific_block(1, 2, 3, tmp_path)
    # Default light_source_type is None -> warning + empty string
    assert out == ""

    with caplog.at_level(logging.WARNING, logger=SIM_MOD_PATH):
        inst.light_source_type = "unknown"
        out = inst._build_source_specific_block(1, 2, 3, tmp_path)
        assert out == ""
        assert any("Unknown light_source_type" in r.message for r in caplog.records)

    inst.light_source_type = "flasher"
    assert inst._build_source_specific_block(1, 2, 3, tmp_path) == "FLASHER"

    inst.light_source_type = "led"
    assert inst._build_source_specific_block(1, 2, 3, tmp_path) == "LED"

    inst.light_source_type = "laser"
    assert inst._build_source_specific_block(1, 2, 3, tmp_path) == "LASER"


def test_make_simtel_script_includes_bypass_for_flasher():
    inst = object.__new__(SimulatorLightEmission)
    inst._logger = logging.getLogger(SIM_MOD_PATH)
    inst.light_source_type = "flasher"
    inst.le_application = ("ff-1m", "layout")
    inst.output_directory = "/directory"

    inst._simtel_path = MagicMock()
    inst._simtel_path.joinpath.return_value = "/path/to/sim_telarray/bin/sim_telarray/"

    inst._telescope_model = MagicMock()
    inst._telescope_model.config_file_directory = "/path/to/config/"
    type(inst._telescope_model).config_file_path = PropertyMock(
        return_value="/path/to/config/config.cfg"
    )
    inst._telescope_model.get_parameter_value.side_effect = (
        lambda p: "atm_test" if p == "atmospheric_transmission" else "X"
    )

    inst._site_model = MagicMock()
    inst._site_model.get_parameter_value_with_unit.return_value = 999 * u.m

    mock_file_content = "dummy"
    with patch("pathlib.Path.open", mock_open(read_data=mock_file_content)):
        cmd = inst._make_simtel_script()

    expected = (
        "SIM_TELARRAY_CONFIG_PATH='' "
        "/path/to/sim_telarray/bin/sim_telarray/ "
        "-I/path/to/config/ -I/path/to/sim_telarray/bin/sim_telarray/ "
        "-c /path/to/config/config.cfg "
        "-DNUM_TELESCOPES=1 "
        "-C altitude=999.0 -C atmospheric_transmission=atm_test "
        "-C TRIGGER_TELESCOPES=1 "
        "-C TELTRIG_MIN_SIGSUM=2 -C PULSE_ANALYSIS=-30 "
        "-C MAXIMUM_TELESCOPES=1 "
        "-C telescope_theta=0 -C telescope_phi=0 "
        "-C Bypass_Optics=1 "
        "-C power_law=2.68 -C input_file=/directory/ff-1m.iact.gz "
        "-C output_file=/directory/ff-1m_layout.simtel.gz "
        "-C histogram_file=/directory/ff-1m_layout.ctsim.hdata\n"
    )

    assert cmd == expected


def test_prepare_ff_atmosphere_files_unlink_ignored_and_copy(tmp_path, monkeypatch):
    inst = object.__new__(SimulatorLightEmission)
    inst._logger = logging.getLogger(SIM_MOD_PATH)
    inst._telescope_model = MagicMock()
    inst._telescope_model.get_parameter_value.return_value = "atm_prof3.dat"

    # Source atmosphere file
    src = tmp_path / "atm_prof3.dat"
    src.write_text("atmcontent3", encoding="utf-8")

    # Pre-create alias files so that unlink() will be attempted on them
    a1 = tmp_path / "atmprof1.dat"
    a2 = tmp_path / "atm_profile_model_1.dat"
    a1.write_text("old1", encoding="utf-8")
    a2.write_text("old2", encoding="utf-8")

    calls = []
    orig_unlink = Path.unlink

    def fake_unlink(self):
        # Record attempts to unlink our aliases and raise OSError to exercise except path
        if self.name in ("atmprof1.dat", "atm_profile_model_1.dat"):
            calls.append(self)
            raise OSError("cannot unlink")
        return orig_unlink(self)

    monkeypatch.setattr(Path, "unlink", fake_unlink, raising=True)

    # Ensure that symlink creation fails so copy fallback is used (since files still exist)
    # In practice, symlink_to on existing path raises FileExistsError, but make it explicit
    def fake_symlink_to(self, target):
        raise OSError("file exists")

    monkeypatch.setattr(Path, "symlink_to", fake_symlink_to, raising=True)

    # Run
    rid = inst._prepare_ff_atmosphere_files(tmp_path)

    # Assertions
    assert rid == 1
    assert len(calls) == 2

    assert a1.exists()
    assert a2.exists()
    # Should end up as regular files with copied content
    assert not a1.is_symlink()
    assert not a2.is_symlink()
    assert a1.read_text(encoding="utf-8") == "atmcontent3"
    assert a2.read_text(encoding="utf-8") == "atmcontent3"


@patch(f"{SIM_MOD_PATH}.SimulatorLightEmission._plot_flasher_outputs")
@patch(f"{SIM_MOD_PATH}.SimulatorLightEmission._plot_calibration_outputs")
@patch(f"{SIM_MOD_PATH}.SimulatorLightEmission._get_distance_for_plotting")
@patch(f"{SIM_MOD_PATH}.SimulatorLightEmission._get_simulation_output_filename")
def test_process_simulation_output_uses_flasher_branch(
    mock_get_filename,
    mock_get_distance,
    mock_plot_calib,
    mock_plot_flasher,
    mock_simulator_variable,
):
    # Arrange
    inst = mock_simulator_variable
    inst.light_source_type = "flasher"
    args_dict = {
        "boundary_thresh": 5,
        "picture_thresh": 3,
        "min_neighbors": 2,
        "return_cleaned": True,
    }
    figures = []
    mock_get_filename.return_value = "dummy.simtel.gz"
    mock_get_distance.return_value = 321 * u.m

    # Act
    inst.process_simulation_output(args_dict, figures)

    # Assert
    mock_plot_flasher.assert_called_once()
    fargs = mock_plot_flasher.call_args[0]
    assert fargs[0] == "dummy.simtel.gz"
    assert fargs[1] == args_dict
    assert fargs[2].to_value(u.m) == pytest.approx(321)
    assert fargs[3] is figures
    mock_plot_calib.assert_not_called()


def test_plot_simulation_output_delegates_to_ctapipe(monkeypatch):
    inst = object.__new__(SimulatorLightEmission)
    captured = {}

    def fake_ctapipe(filename, *, cleaning_args, distance, return_cleaned):
        captured["filename"] = filename
        captured["cleaning_args"] = cleaning_args
        captured["distance"] = distance
        captured["return_cleaned"] = return_cleaned
        return "OK"

    monkeypatch.setattr(sim_mod, "plot_simtel_ctapipe", fake_ctapipe)

    result = inst._plot_simulation_output("out.simtel.gz", 4, 2, 1, 42 * u.m, False)

    assert result == "OK"
    assert captured["filename"] == "out.simtel.gz"
    assert captured["cleaning_args"] == [4, 2, 1]
    assert captured["distance"].to_value(u.m) == pytest.approx(42)
    assert captured["return_cleaned"] is False


def test_prepare_ff_atmosphere_files_warns_on_copy_failure(tmp_path, monkeypatch, caplog):
    # Instance with mocked logger and telescope model
    inst = object.__new__(SimulatorLightEmission)
    inst._logger = logging.getLogger(SIM_MOD_PATH)
    inst._telescope_model = MagicMock()
    inst._telescope_model.get_parameter_value.return_value = "atm_failed.dat"

    # Create source file
    src = tmp_path / "atm_failed.dat"
    src.write_text("atm", encoding="utf-8")

    # Force symlink_to to fail, then copy2 to fail, to trigger warning branch
    monkeypatch.setattr(
        Path,
        "symlink_to",
        lambda self, target: (_ for _ in ()).throw(OSError("no symlink")),
        raising=True,
    )
    monkeypatch.setattr(
        shutil, "copy2", lambda s, d: (_ for _ in ()).throw(OSError("copy failed")), raising=True
    )

    with caplog.at_level(logging.WARNING, logger=SIM_MOD_PATH):
        rid = inst._prepare_ff_atmosphere_files(tmp_path)

    assert rid == 1
    # Two aliases attempted -> two warnings, one per destination name
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("atmprof1.dat" in w and "Failed to create atmosphere alias" in w for w in warnings)
    assert any(
        "atm_profile_model_1.dat" in w and "Failed to create atmosphere alias" in w
        for w in warnings
    )


def test_plot_flasher_outputs_logs_peak_stats(monkeypatch, sim_instance, caplog):
    # Suppress other figures to focus on stats log, but keep calibration fig
    monkeypatch.setattr(sim_mod, "plot_simtel_integrated_signal_image", lambda *a, **k: None)
    monkeypatch.setattr(sim_mod, "plot_simtel_integrated_pedestal_image", lambda *a, **k: None)
    monkeypatch.setattr(sim_mod, "plot_simtel_time_traces", lambda *a, **k: None)
    monkeypatch.setattr(sim_mod, "plot_simtel_waveform_pcolormesh", lambda *a, **k: None)

    # Return a fig and stats dict from peak timing
    stats = {"considered": 10, "found": 8, "mean": 12.34, "std": 1.23}
    monkeypatch.setattr(
        sim_mod, "plot_simtel_peak_timing", lambda *a, **k: (_make_dummy_fig(), stats)
    )

    figures = []
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=SIM_MOD_PATH):
        sim_instance._plot_flasher_outputs("dummy.simtel.gz", {"n_trace_pixels": 4}, None, figures)

    # Expect one calibration fig + one peak timing fig
    assert len(figures) == 2
    messages = "\n".join(r.message for r in caplog.records)
    assert "Peak timing stats:" in messages
    assert "considered=10" in messages
    assert "peaks_found=8" in messages
    assert "mean=12.34" in messages
    assert "std=1.23" in messages
    for f in figures:
        plt.close(f)


def test_photons_per_run_flasher_model_non_test():
    # When flasher model is provided and not in test mode, use model value
    tel = MagicMock()
    tel.write_sim_telarray_config_file = MagicMock()
    flasher = MagicMock()
    flasher.get_parameter_value.return_value = 7.89e6

    inst = SimulatorLightEmission(
        telescope_model=tel,
        calibration_model=None,
        flasher_model=flasher,
        site_model=None,
        light_emission_config={},
        le_application=("ff-1m", "layout"),
        simtel_path=None,
        light_source_type="flasher",
        label="photons-test",
        test=False,
    )

    assert inst.photons_per_run == pytest.approx(7.89e6)
    flasher.get_parameter_value.assert_called_once_with("photons_per_flasher")


def test_photons_per_run_flasher_model_test_mode():
    # When flasher model is provided and in test mode, force 1e8 and don't query model
    tel = MagicMock()
    tel.write_sim_telarray_config_file = MagicMock()
    flasher = MagicMock()

    inst = SimulatorLightEmission(
        telescope_model=tel,
        calibration_model=None,
        flasher_model=flasher,
        site_model=None,
        light_emission_config={},
        le_application=("ff-1m", "layout"),
        simtel_path=None,
        light_source_type="flasher",
        label="photons-test2",
        test=True,
    )

    assert inst.photons_per_run == pytest.approx(1e8)
    flasher.get_parameter_value.assert_not_called()


def test_photons_per_run_no_models():
    # When neither calibration nor flasher model is provided, default to 1e8
    tel = MagicMock()
    tel.write_sim_telarray_config_file = MagicMock()

    inst = SimulatorLightEmission(
        telescope_model=tel,
        calibration_model=None,
        flasher_model=None,
        site_model=None,
        light_emission_config={},
        le_application=("xyzls", "layout"),
        simtel_path=None,
        light_source_type="led",
        label="photons-test3",
        test=False,
    )

    assert inst.photons_per_run == pytest.approx(1e8)

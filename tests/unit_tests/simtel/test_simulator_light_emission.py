#!/usr/bin/python3

import logging
import shutil
import stat
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock, call, mock_open, patch

import astropy.units as u
import numpy as np
import pytest

from simtools.io.io_handler import IOHandler
from simtools.model.calibration_model import CalibrationModel
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.simulator_light_emission import SimulatorLightEmission
from simtools.utils import general as gen

SIM_MOD_PATH = "simtools.simtel.simulator_light_emission"

# Test constants to avoid duplicated string literals
DUMMY_SIMTEL = "dummy.simtel.zst"
ATM_ALIAS1 = "atmprof1.dat"
ATM_ALIAS2 = "atm_profile_model_1.dat"
SIMTEL_BIN = "/path/to/sim_telarray/bin/sim_telarray/"
CONFIG_DIR = "/path/to/config/"
CONFIG_FILE = "/path/to/config/config.cfg"
PATH_OPEN_TARGET = "pathlib.Path.open"
OUT_DIR = "/directory"


def _prepare_inst_with_common_mocks(
    inst,
    *,
    light_source_setup: str = "layout",
    light_source_type: str | None = None,
):
    """Prepare a SimulatorLightEmission instance with common test mocks."""
    inst.light_source_setup = light_source_setup
    inst.output_directory = OUT_DIR
    inst.light_emission_config = {"output_prefix": None, "number_events": 1}

    # Simtel installation path used to compose include paths
    inst._simtel_path = MagicMock()
    inst._simtel_path.joinpath.return_value = SIMTEL_BIN

    # Telescope model provides configuration file and parameters used by sim_telarray
    inst._telescope_model = MagicMock()
    inst._telescope_model.config_file_directory = CONFIG_DIR
    type(inst._telescope_model).config_file_path = PropertyMock(return_value=CONFIG_FILE)

    # Common parameters accessed by code under test
    def _get_tm_param(name):
        if name == "atmospheric_transmission":
            return "atm_test"
        if name == "array_element_position_ground":
            return (1, 1, 1)
        return "X"

    inst._telescope_model.get_parameter_value.side_effect = _get_tm_param

    # Site model supplies altitude (observation level) with units
    inst._site_model = MagicMock()
    inst._site_model.get_parameter_value_with_unit.return_value = 999 * u.m

    # Return concrete values for site params used in simtel script
    def _get_site_param(param):
        if param == "atmospheric_transmission":
            return "atm_test"
        if param == "corsika_observation_level":
            return "999"
        return MagicMock()

    inst._site_model.get_parameter_value.side_effect = _get_site_param

    if light_source_type is not None:
        inst.light_source_type = light_source_type

    return inst


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

    light_source_type = "illuminator"
    return SimulatorLightEmission(
        telescope_model=telescope_model,
        calibration_model=calibration_model,
        site_model=site_model_north,
        light_emission_config={"number_events": 1, "output_prefix": None},
        simtel_path=simtel_path,
        light_source_type=light_source_type,
        light_source_setup="layout",
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

    light_source_type = "illuminator"
    return SimulatorLightEmission(
        telescope_model=telescope_model,
        calibration_model=calibration_model,
        site_model=site_model_north,
        light_emission_config={**default_config, "number_events": 1, "output_prefix": None},
        simtel_path=simtel_path,
        light_source_type=light_source_type,
        light_source_setup="variable",
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
    assert mock_simulator.light_source_type == "illuminator"
    assert mock_simulator.light_emission_config.get("number_events", 1) == 1


def test_initialization_variable(mock_simulator_variable, default_config):
    assert isinstance(mock_simulator_variable, SimulatorLightEmission)
    assert mock_simulator_variable.light_source_type == "illuminator"
    # default config plus CLI-derived settings
    for k, v in default_config.items():
        assert mock_simulator_variable.light_emission_config[k] == v
    assert mock_simulator_variable.light_emission_config.get("number_events", 1) == 1


def test_runs(mock_simulator):
    assert mock_simulator.number_events == 1


def test_photons_per_run_default(mock_simulator):
    assert mock_simulator.photons_per_run == pytest.approx(1e8)


def test_make_light_emission_script(
    mock_simulator,
    site_model_north,
    mock_output_path,
):
    """layout coordinate vector between LST and ILLN"""
    expected_command = (
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

    command = mock_simulator_variable._make_light_emission_script()

    assert command == expected_command

    # removed laser command test; laser mode no longer supported


def test_calibration_pointing_direction(mock_simulator):
    pointing_vector, angles = mock_simulator.calibration_pointing_direction()

    expected_pointing_vector = [0.979, -0.104, -0.174]
    expected_angles = [79.952, 186.092, 79.952, 173.908]

    np.testing.assert_array_almost_equal(pointing_vector, expected_pointing_vector, decimal=3)
    np.testing.assert_array_almost_equal(angles, expected_angles, decimal=3)


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

        # Set paths and config
        mock_simulator._simtel_path.joinpath.return_value = SIMTEL_BIN
        path_to_config_directory = CONFIG_DIR
        mock_simulator._telescope_model.config_file_directory = path_to_config_directory
        path_to_config_file = CONFIG_FILE
        config_file_path_mock = PropertyMock(return_value=path_to_config_file)
        type(mock_simulator._telescope_model).config_file_path = config_file_path_mock

        # Patch Path.open to mock file handling for the config file
        with patch(PATH_OPEN_TARGET, mock_open(read_data=mock_file_content)) as mock_path_open:

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

            def _get_site_param2(param):
                if param == "atmospheric_transmission":
                    return "atm_test"
                if param == "corsika_observation_level":
                    return "999"
                return MagicMock()

            mock_simulator._site_model.get_parameter_value.side_effect = _get_site_param2

            mock_simulator.output_directory = OUT_DIR
            mock_simulator.light_emission_config = {"output_prefix": None, "number_events": 1}

            expected_command = (
                "SIM_TELARRAY_CONFIG_PATH='' "
                f"{SIMTEL_BIN} "
                f"-I{CONFIG_DIR} -I{SIMTEL_BIN} "
                f"-c {CONFIG_FILE} "
                "-DNUM_TELESCOPES=1  "
                "-C altitude=999.0 -C atmospheric_transmission=atm_test "
                "-C TRIGGER_TELESCOPES=1 "
                "-C TELTRIG_MIN_SIGSUM=2 -C PULSE_ANALYSIS=-30 "
                "-C MAXIMUM_TELESCOPES=1 "
                "-C telescope_theta=76.980826 -C telescope_phi=180.17047 "
                f"-C power_law=2.68 -C input_file={OUT_DIR}/xyzls.iact.gz "
                f"-C output_file={OUT_DIR}/xyzls_layout.simtel.zst "
                f"-C histogram_file={OUT_DIR}/xyzls_layout.ctsim.hdata\n"
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
    script_path = mock_simulator.prepare_script()

    assert gen.program_is_executable(script_path)
    mock_make_light_emission_script.assert_called_once()
    mock_make_simtel_script.assert_called_once()

    check_line = (
        f"[ -s '{mock_simulator.output_directory}/{mock_simulator._infer_application()[0]}.iact.gz' ] || "
        "{ echo 'LightEmission did not produce IACT file' >&2; exit 1; }\n\n"
    )
    cleanup_line = f"rm -f '{mock_simulator.output_directory}/{mock_simulator._infer_application()[0]}.iact.gz'\n\n"
    expected_calls = [
        "#!/usr/bin/env bash\n\n",
        "light_emission_script_command\n\n",
        check_line,
        "simtel_script_command\n\n",
        cleanup_line,
        "# End\n\n",
    ]
    for call_args, expected_content in zip(mock_file.write.call_args_list, expected_calls):
        assert call_args[0][0] == expected_content


def test_prepare_script_raises_if_output_exists(mock_simulator, tmp_path):
    mock_simulator.output_directory = tmp_path

    expected_out = tmp_path / "xyzls_layout.simtel.zst"
    expected_out.write_text("", encoding="utf-8")

    with pytest.raises(FileExistsError):
        mock_simulator.prepare_script()


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
def test_simulate_variable_distances(mock_run_simulation, mock_simulator_variable):
    """Test simulating light emission for variable distances."""
    mock_simulator_variable.light_emission_config["z_pos"]["default"] = [100 * u.m, 200 * u.m]
    args_dict = {"distances_ls": None, "telescope": "LSTN-01"}

    mock_simulator_variable.simulate_variable_distances(args_dict)

    assert mock_run_simulation.call_count == 2

    args_dict = {"distances_ls": [100, 200], "telescope": "LSTN-01"}
    mock_simulator_variable.simulate_variable_distances(args_dict)
    assert mock_run_simulation.call_count == 4


@patch(f"{SIM_MOD_PATH}.SimulatorLightEmission.run_simulation")
def test_simulate_layout_positions(mock_run_simulation, mock_simulator):
    """Test simulating light emission for layout positions."""
    args_dict = {"telescope": "LSTN-01"}

    mock_simulator.simulate_layout_positions(args_dict)

    mock_run_simulation.assert_called_once()


def test_get_simulation_output_filename(mock_simulator_variable):
    """Test the _get_simulation_output_filename method."""
    mock_simulator_variable.output_directory = "./tests/resources/"

    filename = mock_simulator_variable._get_simulation_output_filename()

    expected_filename = "./tests/resources//xyzls_variable_d_1000.simtel.zst"
    assert filename == expected_filename


@patch("subprocess.run")
@patch("builtins.open", new_callable=mock_open)
def test_run_simulation(mock_open, mock_subprocess_run, mock_simulator_variable):
    """Test the run_simulation method."""

    mock_simulator_variable.prepare_script = Mock(return_value="dummy_script.sh")
    mock_simulator_variable.run_simulation()

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


def test__build_source_specific_block_unknown_logs_warning(mock_simulator, caplog):
    mock_simulator.light_source_type = "unknown"
    caplog.clear()
    with caplog.at_level("WARNING", logger=mock_simulator._logger.name):
        seg = mock_simulator._build_source_specific_block(0, 0, 0, Path())
        assert seg == ""
        assert any("Unknown light_source_type" in r.message for r in caplog.records)


def test__get_prefix_none_and_value(mock_simulator):
    mock_simulator.light_emission_config["output_prefix"] = None
    assert mock_simulator._get_prefix() == ""
    mock_simulator.light_emission_config["output_prefix"] = "xyzls"
    assert mock_simulator._get_prefix() == "xyzls_"


def test__get_distance_for_plotting_branches(mock_simulator):
    # flasher branch
    mock_simulator.light_source_type = "flasher"
    # Provide a flasher model mock since the fixture doesn't include one
    mock_simulator._flasher_model = MagicMock()
    mock_simulator._flasher_model.get_parameter_value_with_unit.return_value = 60 * u.cm
    assert mock_simulator._get_distance_for_plotting().to(u.cm).value == 60

    # variable z_pos dict with list default
    mock_simulator.light_source_type = "illuminator"
    mock_simulator.light_emission_config["z_pos"] = {"default": [100 * u.m]}
    assert mock_simulator._get_distance_for_plotting().to_value(u.m) == 100

    # fallback to instance.distance
    mock_simulator.light_emission_config.pop("z_pos")
    mock_simulator.distance = 42 * u.m
    assert mock_simulator._get_distance_for_plotting().to_value(u.m) == 42

    # fallback to instance.distance
    mock_simulator.distance = 42
    assert mock_simulator._get_distance_for_plotting().to_value(u.m) == 42

    # final fallback to 0 m
    mock_simulator.distance = None
    assert mock_simulator._get_distance_for_plotting().to_value(u.m) == 0


def test_distance_list_valid_and_error(mock_simulator):
    vals = mock_simulator.distance_list(["1", 2, 3.5])
    assert [v.to_value(u.m) for v in vals] == [1.0, 2.0, 3.5]
    with pytest.raises(ValueError, match="numeric"):
        mock_simulator.distance_list(["a", 1])


def test_update_light_emission_config_keyerror(mock_simulator):
    with pytest.raises(KeyError):
        mock_simulator.update_light_emission_config("nope", 1)


def test__build_altitude_atmo_block_flasher(mock_simulator, tmp_path, monkeypatch):
    mock_simulator._simtel_path = Path("/opt/simtel")
    monkeypatch.setattr(mock_simulator, "_prepare_flasher_atmosphere_files", lambda d: 1)
    seg = mock_simulator._build_altitude_atmo_block(
        "ff-1m",
        tmp_path,
        1000 * u.m,
        tmp_path / "telpos.dat",
    )
    assert "--altitude 1000.0" in seg
    assert "--atmosphere 1" in seg
    assert f"-I{mock_simulator._simtel_path.joinpath('sim_telarray/cfg')}" in seg


def test__prepare_flasher_atmosphere_files_creates_aliases(mock_simulator, tmp_path):
    src_name = "atm_src_profile.dat"
    (tmp_path / src_name).write_text("x", encoding="utf-8")
    # Ensure telescope model is a mock for this test
    mock_simulator._telescope_model = MagicMock()
    mock_simulator._site_model = MagicMock()
    mock_simulator._site_model.get_parameter_value.return_value = src_name
    atmo_id = mock_simulator._prepare_flasher_atmosphere_files(tmp_path)
    assert atmo_id == 1
    assert (tmp_path / "atmprof1.dat").exists()
    assert (tmp_path / "atm_profile_model_1.dat").exists()


def test_prepare_script_raises_if_output_exists_dup_removed():
    # Removed duplicated test to avoid redefinition; original test covers this case.
    pass


def test_run_simulation_warns_when_no_output(mock_simulator, tmp_path, monkeypatch, caplog):
    # Create a tiny runnable script file
    script = tmp_path / "run.sh"
    script.write_text("#!/usr/bin/env bash\necho hi\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)

    monkeypatch.setattr(mock_simulator, "prepare_script", lambda: script)
    target_out = tmp_path / "target.simtel.zst"
    if target_out.exists():
        target_out.unlink()
    mock_simulator._get_simulation_output_filename = lambda: str(target_out)

    caplog.clear()
    with caplog.at_level("WARNING", logger=mock_simulator._logger.name):
        out_path = mock_simulator.run_simulation()
    assert Path(out_path) == target_out
    assert any("Expected simtel output not found" in r.message for r in caplog.records)


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


def test_add_flasher_options():
    inst = object.__new__(SimulatorLightEmission)
    inst._flasher_model = MagicMock()
    inst._telescope_model = MagicMock()
    inst.number_events = 1
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


def test_prepare_flasher_atmosphere_files_creates_aliases(tmp_path, caplog):
    inst = object.__new__(SimulatorLightEmission)
    inst._logger = logging.getLogger(SIM_MOD_PATH)
    inst._site_model = MagicMock()
    inst._site_model.get_parameter_value.return_value = "atm_test_profile.dat"

    src = tmp_path / "atm_test_profile.dat"
    src.write_text("atmcontent", encoding="utf-8")

    with caplog.at_level(logging.DEBUG, logger=SIM_MOD_PATH):
        rid = inst._prepare_flasher_atmosphere_files(tmp_path)

    assert rid == 1
    for name in (ATM_ALIAS1, ATM_ALIAS2):
        alias = tmp_path / name
        assert alias.exists()
        # Content must match source (works for symlink or copied file)
        assert alias.read_text(encoding="utf-8") == "atmcontent"


def test_prepare_flasher_atmosphere_files_copy_fallback(tmp_path, monkeypatch):
    inst = object.__new__(SimulatorLightEmission)
    inst._logger = logging.getLogger(SIM_MOD_PATH)
    inst._site_model = MagicMock()
    inst._site_model.get_parameter_value.return_value = "atm_prof2.dat"

    src = tmp_path / "atm_prof2.dat"
    src.write_text("X", encoding="utf-8")

    def raise_oserr(self, target):
        raise OSError("symlink not permitted")

    monkeypatch.setattr(Path, "symlink_to", raise_oserr, raising=True)

    def fake_copy2(src_path, dst_path):
        Path(dst_path).write_bytes(Path(src_path).read_bytes())
        return str(dst_path)

    monkeypatch.setattr(shutil, "copy2", fake_copy2, raising=True)

    rid = inst._prepare_flasher_atmosphere_files(tmp_path)
    assert rid == 1

    a1 = tmp_path / ATM_ALIAS1
    a2 = tmp_path / ATM_ALIAS2
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

    monkeypatch.setattr(inst, "_prepare_flasher_atmosphere_files", lambda *_: 42)

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
    monkeypatch.setattr(inst, "_add_illuminator_command_options", lambda cmd: "LED")
    monkeypatch.setattr(
        inst,
        "_add_flasher_command_options",
        lambda cmd: "FLASHER",
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

    inst.light_source_type = "illuminator"
    assert inst._build_source_specific_block(1, 2, 3, tmp_path) == "LED"

    inst.light_source_type = "flasher"
    assert inst._build_source_specific_block(1, 2, 3, tmp_path) == "FLASHER"


def test_make_simtel_script_includes_bypass_for_flasher():
    inst = object.__new__(SimulatorLightEmission)
    inst._logger = logging.getLogger(SIM_MOD_PATH)
    _prepare_inst_with_common_mocks(inst, light_source_type="flasher")

    mock_file_content = "dummy"
    with patch(PATH_OPEN_TARGET, mock_open(read_data=mock_file_content)):
        cmd = inst._make_simtel_script()

    expected = (
        "SIM_TELARRAY_CONFIG_PATH='' "
        f"{SIMTEL_BIN} "
        f"-I{CONFIG_DIR} -I{SIMTEL_BIN} "
        f"-c {CONFIG_FILE} "
        "-DNUM_TELESCOPES=1  "
        "-C altitude=999.0 -C atmospheric_transmission=atm_test "
        "-C TRIGGER_TELESCOPES=1 "
        "-C TELTRIG_MIN_SIGSUM=2 -C PULSE_ANALYSIS=-30 "
        "-C MAXIMUM_TELESCOPES=1 "
        "-C telescope_theta=0 -C telescope_phi=0 "
        "-C Bypass_Optics=1 "
        f"-C power_law=2.68 -C input_file={OUT_DIR}/ff-1m.iact.gz "
        f"-C output_file={OUT_DIR}/ff-1m_flasher.simtel.zst "
        f"-C histogram_file={OUT_DIR}/ff-1m_flasher.ctsim.hdata\n"
    )

    assert cmd == expected


def test_prepare_flasher_atmosphere_files_unlink_ignored_and_copy(tmp_path, monkeypatch):
    inst = object.__new__(SimulatorLightEmission)
    inst._logger = logging.getLogger(SIM_MOD_PATH)
    inst._site_model = MagicMock()
    inst._site_model.get_parameter_value.return_value = "atm_prof3.dat"

    # Source atmosphere file
    src = tmp_path / "atm_prof3.dat"
    src.write_text("atmcontent3", encoding="utf-8")

    # Pre-create alias files so that unlink() will be attempted on them
    a1 = tmp_path / ATM_ALIAS1
    a2 = tmp_path / ATM_ALIAS2
    a1.write_text("old1", encoding="utf-8")
    a2.write_text("old2", encoding="utf-8")

    calls = []
    orig_unlink = Path.unlink

    def fake_unlink(self):
        # Record attempts to unlink our aliases and raise OSError to exercise except path
        if self.name in (ATM_ALIAS1, ATM_ALIAS2):
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
    rid = inst._prepare_flasher_atmosphere_files(tmp_path)

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


def test_prepare_flasher_atmosphere_files_warns_on_copy_failure(tmp_path, monkeypatch, caplog):
    # Instance with mocked logger and telescope model
    inst = object.__new__(SimulatorLightEmission)
    inst._logger = logging.getLogger(SIM_MOD_PATH)
    inst._site_model = MagicMock()
    inst._site_model.get_parameter_value.return_value = "atm_failed.dat"

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
        rid = inst._prepare_flasher_atmosphere_files(tmp_path)

    assert rid == 1
    # Two aliases attempted -> two warnings, one per destination name
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any(ATM_ALIAS1 in w and "Failed to create atmosphere alias" in w for w in warnings)
    assert any((ATM_ALIAS2 in w and "Failed to create atmosphere alias" in w) for w in warnings)


def test_photons_per_run_flasher_model_non_test(tmp_path):
    # When flasher model is provided and not in test mode, use model value
    IOHandler().set_paths(
        output_path=str(tmp_path), data_path=str(tmp_path), model_path=str(tmp_path)
    )
    tel = MagicMock()
    tel.write_sim_telarray_config_file = MagicMock()
    flasher = MagicMock()
    flasher.get_parameter_value.return_value = 7.89e6

    inst = SimulatorLightEmission(
        telescope_model=tel,
        calibration_model=None,
        flasher_model=flasher,
        site_model=None,
        light_emission_config={"number_events": 1, "output_prefix": None},
        simtel_path=tmp_path,
        light_source_type="flasher",
        label="photons-test",
        test=False,
    )

    assert inst.photons_per_run == pytest.approx(7.89e6)
    flasher.get_parameter_value.assert_called_once_with("photons_per_flasher")


def test_photons_per_run_flasher_model_test_mode(tmp_path):
    IOHandler().set_paths(
        output_path=str(tmp_path), data_path=str(tmp_path), model_path=str(tmp_path)
    )
    tel = MagicMock()
    tel.write_sim_telarray_config_file = MagicMock()
    flasher = MagicMock()

    inst = SimulatorLightEmission(
        telescope_model=tel,
        calibration_model=None,
        flasher_model=flasher,
        site_model=None,
        light_emission_config={"number_events": 1, "output_prefix": None},
        simtel_path=tmp_path,
        light_source_type="flasher",
        label="photons-test2",
        test=True,
    )

    assert inst.photons_per_run == pytest.approx(1e8)
    flasher.get_parameter_value.assert_not_called()


def test_photons_per_run_no_models(tmp_path):
    # When neither calibration nor flasher model is provided, default to 1e8
    tel = MagicMock()
    tel.write_sim_telarray_config_file = MagicMock()

    inst = SimulatorLightEmission(
        telescope_model=tel,
        calibration_model=None,
        flasher_model=None,
        site_model=None,
        light_emission_config={"number_events": 1, "output_prefix": None},
        simtel_path=tmp_path,
        light_source_type="illuminator",
        label="photons-test3",
        test=False,
    )

    assert inst.photons_per_run == pytest.approx(1e8)


def test_get_prefix_non_none_returns_with_underscore():
    inst = object.__new__(SimulatorLightEmission)
    inst.light_emission_config = {"output_prefix": "pre", "number_events": 1}
    assert inst._get_prefix() == "pre_"


def test_make_simtel_script_variable_type_sets_zero_angles():
    inst = object.__new__(SimulatorLightEmission)
    inst._logger = logging.getLogger(SIM_MOD_PATH)
    _prepare_inst_with_common_mocks(inst)
    inst.light_source_type = "variable"

    # Avoid calling real calibration method
    inst.calibration_pointing_direction = MagicMock(return_value=([0, 0, 1], [10, 20]))

    mock_file_content = "dummy"
    with patch(PATH_OPEN_TARGET, mock_open(read_data=mock_file_content)):
        cmd = inst._make_simtel_script()

    assert "-C telescope_theta=0 -C telescope_phi=0 " in cmd


def test_make_simtel_script_variable_dist_suffix_exception():
    inst = object.__new__(SimulatorLightEmission)
    inst._logger = logging.getLogger(SIM_MOD_PATH)
    _prepare_inst_with_common_mocks(inst)
    inst.light_source_type = "illuminator"
    inst.light_source_setup = "variable"

    # Force exception path for distance suffix
    inst._get_distance_for_plotting = MagicMock(side_effect=Exception("boom"))
    # Avoid calling real calibration method
    inst.calibration_pointing_direction = MagicMock(return_value=([0, 0, 1], [10, 20]))

    mock_file_content = "dummy"
    with patch(PATH_OPEN_TARGET, mock_open(read_data=mock_file_content)):
        cmd = inst._make_simtel_script()

    assert "-C output_file=/directory/xyzls_variable.simtel.zst " in cmd
    assert "-C histogram_file=/directory/xyzls_variable.ctsim.hdata\n" in cmd


def test_get_simulation_output_filename_prefix_and_exception():
    inst = object.__new__(SimulatorLightEmission)
    inst._logger = logging.getLogger(SIM_MOD_PATH)
    inst.output_directory = "/out"
    inst.light_source_type = "illuminator"
    inst.light_source_setup = "variable"
    inst.light_emission_config = {"output_prefix": "pre", "number_events": 1}

    # Cause exception so no distance suffix is appended
    inst._get_distance_for_plotting = MagicMock(side_effect=Exception("err"))

    # Use real inference for app name/mode
    def infer():
        return ("xyzls", "variable")

    inst._infer_application = infer

    out = inst._get_simulation_output_filename()
    assert out == "/out/pre_xyzls_variable.simtel.zst"


def test_light_emission_default_configuration_schema():
    cfg = SimulatorLightEmission.light_emission_default_configuration()
    # Basic type and required keys
    assert isinstance(cfg, dict)
    for key in ("zenith_angle", "azimuth_angle", "source_distance", "off_axis_angle", "fadc_bins"):
        assert key in cfg
        assert "len" in cfg[key]
        assert "default" in cfg[key]
        assert "names" in cfg[key]
    # Units
    assert cfg["zenith_angle"]["unit"] == u.deg
    assert cfg["azimuth_angle"]["unit"] == u.deg
    assert cfg["source_distance"]["unit"] == u.m
    assert cfg["off_axis_angle"]["unit"] == u.deg
    # Defaults have expected types
    assert cfg["zenith_angle"]["default"].unit == u.deg
    assert cfg["source_distance"]["default"].unit == u.m


def test_flasher_default_configuration_schema():
    cfg = SimulatorLightEmission.flasher_default_configuration()
    assert isinstance(cfg, dict)
    for key in (
        "number_events",
        "photons_per_flasher",
        "bunch_size",
        "flasher_position",
        "flasher_depth",
        "flasher_inclination",
        "spectrum",
        "lightpulse",
        "angular_distribution",
        "flasher_pattern",
    ):
        assert key in cfg
        assert "len" in cfg[key]
        assert "default" in cfg[key]
        assert "names" in cfg[key]

    # Vector and units checks
    pos = cfg["flasher_position"]
    assert pos["len"] == 2
    assert pos["unit"] == u.cm
    assert len(pos["default"]) == 2
    assert pos["default"].unit == u.cm

    assert cfg["flasher_depth"]["unit"] == u.cm
    assert cfg["flasher_inclination"]["unit"] == u.deg
    assert cfg["spectrum"]["unit"] == u.nm

import logging
import mmap
import os
import re
import tarfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest import mock
from unittest.mock import PropertyMock

import matplotlib.pyplot as plt
import pytest
from astropy import units as u
from dotenv import load_dotenv

import simtools.io.io_handler
from simtools import settings
from simtools.configuration.configurator import Configurator
from simtools.corsika.corsika_config import CorsikaConfig
from simtools.db import db_handler
from simtools.model.array_model import ArrayModel
from simtools.model.site_model import SiteModel
from simtools.model.telescope_model import TelescopeModel
from simtools.runners.corsika_runner import CorsikaRunner

logger = logging.getLogger()


def pytest_addoption(parser):
    """Model version command line parameter."""
    parser.addoption("--model_version", action="store", default=None)


@pytest.fixture(autouse=True)
def simtools_settings(tmp_test_directory, db_config):
    """Load simtools settings for the test session."""
    load_dotenv(".env")
    settings.config.load(db_config=db_config)


@pytest.fixture(autouse=True)
def mock_simulator_paths(request, tmp_test_directory):
    """Mock sim_telarray and corsika paths for unit tests.

    This fixture mocks the paths and executables to prevent FileNotFoundError
    when sim_telarray or corsika are not installed in the test environment.

    This fixture does NOT apply to:

    - test_settings.py (tests path validation behavior)
    - integration_tests (need full installations with CORSIKA and sim_telarray to run)
    """
    test_file_path = str(request.node.fspath)
    if "test_settings.py" in test_file_path or "integration_tests" in test_file_path:
        yield
        return

    # Create mock directories
    sim_telarray_dir = Path(tmp_test_directory) / "sim_telarray"
    corsika_dir = Path(tmp_test_directory) / "corsika"
    interaction_table_dir = Path(tmp_test_directory) / "corsika_interaction_tables"

    for directory in [sim_telarray_dir / "bin", corsika_dir, interaction_table_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    # Define mock property paths
    mock_paths = {
        "sim_telarray_path": sim_telarray_dir,
        "corsika_path": corsika_dir,
        "sim_telarray_exe": sim_telarray_dir / "bin" / "sim_telarray",
        "sim_telarray_exe_debug_trace": sim_telarray_dir / "bin" / "sim_telarray_debug_trace",
        "corsika_exe": corsika_dir / "corsika_flat",
        "corsika_exe_curved": corsika_dir / "corsika_curved",
        "corsika_interaction_table_path": interaction_table_dir,
    }

    # Apply patches for all properties
    with ExitStack() as stack:
        for prop_name, return_value in mock_paths.items():
            mock_obj = stack.enter_context(
                mock.patch.object(type(settings.config), prop_name, new_callable=PropertyMock)
            )
            mock_obj.return_value = return_value
        yield


@pytest.fixture(scope="session", autouse=True)
def _set_matplotlib_backend():
    """Set matplotlib backend to Agg for testing (plotting without display server)."""
    plt.switch_backend("Agg")


@pytest.fixture
def tmp_test_directory(tmpdir_factory):
    """Sets temporary test directories. Some tests depend on this structure."""

    tmp_test_dir = tmpdir_factory.mktemp("test-data")
    tmp_sub_dirs = ["resources", "output", "sim_telarray", "model", "application-plots"]
    for sub_dir in tmp_sub_dirs:
        tmp_sub_dir = tmp_test_dir / sub_dir
        tmp_sub_dir.mkdir()

    return tmp_test_dir


@pytest.fixture
def data_path():
    return "./data/"


@pytest.fixture(autouse=True)
def io_handler(tmp_test_directory, data_path):
    """Define io_handler fixture including output and model directories."""
    tmp_io_handler = simtools.io.io_handler.IOHandler()
    tmp_io_handler.set_paths(
        output_path=str(tmp_test_directory) + "/output",
        model_path=str(tmp_test_directory) + "/model",
    )
    return tmp_io_handler


@pytest.fixture
def _mock_settings_env_vars(tmp_test_directory):
    """Removes all environment variable from the test system and explicitly sets those needed."""

    with mock.patch.dict(
        os.environ,
        {
            "SIMTOOLS_SIM_TELARRAY_PATH": str(settings.config.sim_telarray_path),
            "SIMTOOLS_DB_API_USER": "db_user",
            "SIMTOOLS_DB_API_PW": "12345",
            "SIMTOOLS_DB_API_PORT": "42",
            "SIMTOOLS_DB_SERVER": "abc@def.de",
            "SIMTOOLS_DB_SIMULATION_MODEL": "sim_model",
            "SIMTOOLS_DB_SIMULATION_MODEL_VERSION": "v0.0.0",
        },
        clear=True,
    ):
        yield


@pytest.fixture
def args_dict(tmp_test_directory, data_path):
    """Minimal configuration from command line."""
    return Configurator().default_config(
        (
            "--output_path",
            str(tmp_test_directory),
            "--data_path",
            data_path,
        ),
    )


@pytest.fixture
def args_dict_site(tmp_test_directory, data_path):
    "Configuration include site and telescopes."
    return Configurator().default_config(
        (
            "--output_path",
            str(tmp_test_directory),
            "--data_path",
            data_path,
            "--site",
            "South",
            "--telescope",
            "MSTS-07",
            "--label",
            "integration_test",
        )
    )


@pytest.fixture(scope="session", autouse=True)
def mongo_db_logger_settings():
    """Suppress MongoDB 'IdleConnectionMonitor' DEBUG logs during tests."""
    monitor_logger = logging.getLogger("IdleConnectionMonitor")
    monitor_logger.setLevel(logging.INFO)
    logger.info("[TEST SETUP] Suppressing MongoDB 'IdleConnectionMonitor' DEBUG logs.")


@pytest.fixture
def db_config():
    """DB configuration from .env file."""
    load_dotenv(".env")

    _db_para = (
        "db_api_user",
        "db_api_pw",
        "db_api_port",
        "db_api_authentication_database",
        "db_server",
        "db_simulation_model",
        "db_simulation_model_version",
    )
    db_config = {_para: os.environ.get(f"SIMTOOLS_{_para.upper()}") for _para in _db_para}
    if db_config["db_api_port"] is not None:
        db_config["db_api_port"] = int(db_config["db_api_port"])
    return db_config


@pytest.fixture
def mock_db_handler():
    """
    Mock DatabaseHandler for unit tests.

    Provides common mock behaviors to avoid real database connections.
    Returns a MagicMock configured with typical DatabaseHandler methods.
    """
    from unittest.mock import MagicMock

    # Minimal mock parameters for common telescope model tests
    mock_parameters = {
        "num_gains": {
            "value": 2,
            "parameter_version": "1.0.0",
            "type": "int64",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "camera_pixels": {
            "value": 1855,
            "parameter_version": "1.0.0",
            "type": "int32",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "mirror_focal_length": {
            "value": 28.0,
            "parameter_version": "1.0.0",
            "type": "float64",
            "unit": "m",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "mirror_reflection_random_angle": {
            "value": [0.0066, 0.0],
            "parameter_version": "1.0.0",
            "type": "float64",
            "unit": "deg",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "telescope_axis_height": {
            "value": 16.0,
            "parameter_version": "1.0.0",
            "type": "float64",
            "unit": "m",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "telescope_transmission": {
            "value": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "parameter_version": "1.0.0",
            "type": "float64",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "telescope_sphere_radius": {
            "value": 15.0,
            "parameter_version": "1.0.0",
            "type": "float64",
            "unit": "m",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "focal_length": {
            "value": 1600.0,
            "parameter_version": "1.0.0",
            "type": "float64",
            "unit": "cm",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "mirror_list": {
            "value": "mirror_list_dummy.dat",
            "parameter_version": "1.0.0",
            "type": "str",
            "file": True,
            "model_parameter_schema_version": "1.0.0",
        },
        "mirror_class": {
            "value": 0,
            "parameter_version": "1.0.0",
            "type": "uint64",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        # Site parameters (site-specific ones like reference_point_* are in site_specific_params)
        "array_layouts": {
            "value": "test_layout",
            "parameter_version": "1.0.0",
            "type": "str",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "geomag_horizontal": {
            "value": 20.0,
            "parameter_version": "1.0.0",
            "type": "float64",
            "unit": "uT",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "geomag_vertical": {
            "value": -10.0,
            "parameter_version": "1.0.0",
            "type": "float64",
            "unit": "uT",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "geomag_rotation": {
            "value": 0.0,
            "parameter_version": "1.0.0",
            "type": "float64",
            "unit": "deg",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "nsb_reference_spectrum": {
            "value": "nsb_spectrum_dummy.dat",
            "parameter_version": "1.0.0",
            "type": "str",
            "file": True,
            "model_parameter_schema_version": "1.0.0",
        },
        "nsb_reference_value": {
            "value": 0.24,
            "parameter_version": "1.0.0",
            "type": "float64",
            "unit": "1/(sr*ns*cm**2)",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "camera_config_file": {
            "value": "camera_config_dummy.dat",
            "parameter_version": "1.0.0",
            "type": "str",
            "file": True,
            "model_parameter_schema_version": "1.0.0",
        },
        "effective_focal_length": {
            "value": 28.0,
            "parameter_version": "1.0.0",
            "type": "float64",
            "unit": "m",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "optics_properties": {
            "value": "optics_properties_dummy.dat",
            "parameter_version": "1.0.0",
            "type": "str",
            "file": True,
            "model_parameter_schema_version": "1.0.0",
        },
        "camera_filter_incidence_angle": {
            "value": "camera_filter_incidence_angle_dummy.dat",
            "parameter_version": "1.0.0",
            "type": "str",
            "file": True,
            "model_parameter_schema_version": "1.0.0",
        },
        "parabolic_dish": {
            "value": True,
            "parameter_version": "1.0.0",
            "type": "bool",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "camera_filter": {
            "value": "camera_filter_dummy.dat",
            "parameter_version": "1.0.0",
            "type": "str",
            "file": True,
            "model_parameter_schema_version": "1.0.0",
        },
        "dish_shape_length": {
            "value": 5.56,
            "parameter_version": "1.0.0",
            "type": "double",
            "unit": "m",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "camera_transmission": {
            "value": "camera_transmission_dummy.dat",
            "parameter_version": "1.0.0",
            "type": "str",
            "file": True,
            "model_parameter_schema_version": "1.0.0",
        },
        "mirror_reflectivity": {
            "value": "mirror_reflectivity_dummy.dat",
            "parameter_version": "1.0.0",
            "type": "str",
            "file": True,
            "model_parameter_schema_version": "1.0.0",
        },
        "quantum_efficiency": {
            "value": "quantum_efficiency_dummy.dat",
            "parameter_version": "1.0.0",
            "type": "str",
            "file": True,
            "model_parameter_schema_version": "1.0.0",
        },
    }

    # Mock simulation configuration parameters
    mock_sim_config_params = {
        "correct_nsb_spectrum_to_telescope_altitude": {
            "value": "nsb_spectrum_dummy.dat",
            "parameter_version": "1.0.0",
            "file": True,
            "type": "str",
            "model_parameter_schema_version": "1.0.0",
        },
    }

    # Site-specific parameters (different for North and South)
    site_specific_params_north = {
        "reference_point_utm_east": {
            "value": 217611.227,
            "parameter_version": "1.0.0",
            "type": "float64",
            "unit": "m",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "reference_point_utm_north": {
            "value": 3185066.278,
            "parameter_version": "1.0.0",
            "type": "float64",
            "unit": "m",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "reference_point_altitude": {
            "value": 2156.0,
            "parameter_version": "1.0.0",
            "type": "float64",
            "unit": "m",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "epsg_code": {
            "value": 32628,
            "parameter_version": "1.0.0",
            "type": "int32",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "corsika_observation_level": {
            "value": 2158.0,
            "parameter_version": "1.0.0",
            "type": "float64",
            "unit": "m",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "atmospheric_transmission": {
            "value": "atmospheric_transmission_north_dummy.dat",
            "parameter_version": "1.0.0",
            "type": "str",
            "file": True,
            "model_parameter_schema_version": "1.0.0",
        },
        "atmospheric_profile": {
            "value": "atmospheric_profile_north_dummy.dat",
            "parameter_version": "1.0.0",
            "type": "str",
            "file": True,
            "model_parameter_schema_version": "1.0.0",
        },
    }

    site_specific_params_south = {
        "reference_point_utm_east": {
            "value": 366822.0,
            "parameter_version": "1.0.0",
            "type": "float64",
            "unit": "m",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "reference_point_utm_north": {
            "value": 7269466.0,
            "parameter_version": "1.0.0",
            "type": "float64",
            "unit": "m",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "reference_point_altitude": {
            "value": 2162.0,
            "parameter_version": "1.0.0",
            "type": "float64",
            "unit": "m",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "epsg_code": {
            "value": 32719,
            "parameter_version": "1.0.0",
            "type": "int32",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "corsika_observation_level": {
            "value": 2147.0,
            "parameter_version": "1.0.0",
            "type": "float64",
            "unit": "m",
            "file": False,
            "model_parameter_schema_version": "1.0.0",
        },
        "atmospheric_transmission": {
            "value": "atmospheric_transmission_south_dummy.dat",
            "parameter_version": "1.0.0",
            "type": "str",
            "file": True,
            "model_parameter_schema_version": "1.0.0",
        },
        "atmospheric_profile": {
            "value": "atmospheric_profile_south_dummy.dat",
            "parameter_version": "1.0.0",
            "type": "str",
            "file": True,
            "model_parameter_schema_version": "1.0.0",
        },
    }

    def mock_get_model_parameters(site, array_element_name, collection, model_version):
        """Return site-specific parameters based on the site."""
        params = dict(mock_parameters)
        # Override with site-specific values
        if site == "North":
            params.update(site_specific_params_north)
        elif site == "South":
            params.update(site_specific_params_south)
        return params

    def mock_get_model_parameter(parameter, parameter_version, **kwargs):
        """
        Mock get_model_parameter to return parameters only if they exist with the exact version.

        This allows tests to check for parameter existence in the DB.
        """
        # Get site from kwargs if provided
        site = kwargs.get("site")
        params = dict(mock_parameters)

        # Override with site-specific values
        if site == "North":
            params.update(site_specific_params_north)
        elif site == "South":
            params.update(site_specific_params_south)

        if parameter in params:
            param_data = params[parameter]
            if param_data.get("parameter_version") == parameter_version:
                return param_data
        # If parameter doesn't exist or version doesn't match, raise ValueError
        raise ValueError(f"Parameter {parameter} with version {parameter_version} not found")

    mock_db = MagicMock()
    mock_db.is_configured.return_value = True
    mock_db.get_design_model.return_value = None
    mock_db.get_model_parameters.side_effect = mock_get_model_parameters
    mock_db.get_model_parameter.side_effect = mock_get_model_parameter
    mock_db.get_model_parameters_for_all_model_versions.return_value = {}
    mock_db.get_model_versions.return_value = ["6.0.2", "5.0.0"]
    mock_db.get_array_elements.return_value = ["LSTN-01", "MSTS-01"]
    mock_db.get_simulation_configuration_parameters.return_value = mock_sim_config_params
    mock_db.export_model_files.return_value = None
    mock_db.export_model_file.return_value = None
    mock_db.db_name = "test_db"

    return mock_db


@pytest.fixture(autouse=True)
def mock_database_handler(request, mock_db_handler, mocker):
    """
    Automatically mock DatabaseHandler for all unit tests except those in db/.

    This prevents unit tests from trying to connect to real databases.
    Tests in tests/unit_tests/db/ are excluded from this mocking.
    Also mocks model parameter schema validation to avoid schema version mismatches.
    """
    test_file_path = str(request.node.fspath)

    # Skip mocking for tests in db/ directory
    if "unit_tests/db/" in test_file_path:
        yield
        return

    # Mock schema validation to avoid version check issues
    mocker.patch("simtools.model.model_parameter.ModelParameter._check_model_parameter_versions")

    # Mock DatabaseHandler for all other unit tests
    with mock.patch("simtools.db.db_handler.DatabaseHandler", return_value=mock_db_handler):
        yield


@pytest.fixture
def db():
    """Database object with configuration from settings.config.db_handler."""
    return db_handler.DatabaseHandler()


@pytest.fixture
def model_version():
    """Simulation model version used in tests."""
    return "6.0.2"


@pytest.fixture
def model_version_prod5():
    """Simulation model version used in tests."""
    return "5.0.0"


@pytest.fixture
def array_model_north(io_handler, model_version):
    """Array model for North site."""
    return ArrayModel(
        label="test-lst-array",
        site="North",
        layout_name="test_layout",
        model_version=model_version,
    )


@pytest.fixture
def array_model_south(io_handler, model_version):
    """Array model for South site."""
    return ArrayModel(
        label="test-lst-array",
        site="South",
        layout_name="test_layout",
        model_version=model_version,
    )


@pytest.fixture
def site_model_south(model_version):
    """Site model for South site."""
    return SiteModel(
        site="South",
        label="site-south",
        model_version=model_version,
    )


@pytest.fixture
def site_model_north(model_version):
    """Site model for North site."""
    return SiteModel(
        site="North",
        label="site-north",
        model_version=model_version,
    )


@pytest.fixture
def telescope_model_lst(io_handler, model_version):
    """Telescope model LST North."""
    return TelescopeModel(
        site="North",
        telescope_name="LSTN-01",
        model_version=model_version,
        label="test-telescope-model-lst",
    )


@pytest.fixture
def telescope_model_mst(io_handler, model_version):
    """Telescope model MST FlashCam."""
    return TelescopeModel(
        site="South",
        telescope_name="MSTx-FlashCam",
        model_version=model_version,
        label="test-telescope-model-mst",
    )


@pytest.fixture
def telescope_model_sst(io_handler, model_version):
    """Telescope model SST South."""
    return TelescopeModel(
        site="South",
        telescope_name="SSTS-design",
        model_version=model_version,
        label="test-telescope-model-sst",
    )


# keep 5.0.0 model until a complete prod6 model is in the DB
@pytest.fixture
def telescope_model_sst_prod5(io_handler, model_version_prod5):
    """Telescope model SST South (prod5/5.0.0)."""
    return TelescopeModel(
        site="South",
        telescope_name="SSTS-design",
        model_version=model_version_prod5,
        label="test-telescope-model-sst",
    )


@pytest.fixture
def telescope_north_with_calibration_devices_test_file():
    """Telescope positions North with calibration devices."""
    return "tests/resources/telescope_positions-North-with-calibration-devices-ground.ecsv"


@pytest.fixture
def telescope_north_test_file():
    """Telescope positions North."""
    return "tests/resources/telescope_positions-North-ground.ecsv"


@pytest.fixture
def telescope_north_utm_test_file():
    """Telescope positions North (UTM coordinates)."""
    return "tests/resources/telescope_positions-North-utm.ecsv"


@pytest.fixture
def telescope_south_test_file():
    """Telescope positions South."""
    return "tests/resources/telescope_positions-South-ground.ecsv"


@pytest.fixture
def corsika_config_data(model_version):
    """Corsika configuration data (as given by CorsikaConfig)."""
    return {
        "nshow": 100,
        "run_number_offset": 0,
        "run_number": 1,
        "event_number_first_shower": 1,
        "zenith_angle": 20 * u.deg,
        "azimuth_angle": 0.0 * u.deg,
        "view_cone": (0.0 * u.deg, 10.0 * u.deg),
        "energy_range": (10.0 * u.GeV, 10.0 * u.TeV),
        "eslope": -2,
        "core_scatter": (10, 1400.0 * u.m),
        "primary": "proton",
        "primary_id_type": "common_name",
        "correct_for_b_field_alignment": True,
        "model_version": model_version,
    }


@pytest.fixture
def corsika_config_mock_array_model(corsika_config_data, model_version):
    """Corsika configuration object (using array model South)."""
    array_model = mock.MagicMock()
    array_model.layout_name = "test_layout"
    array_model.corsika_config.primary = "proton"
    array_model.site_model = mock.MagicMock()
    array_model.site_model._parameters = {
        "geomag_rotation": -4.533,
        "corsika_observation_level": 2200.0,
    }
    array_model.model_version = model_version

    # Define get_parameter_value() to behave as expected
    def mock_get_parameter_value(par_name):
        return array_model.site_model._parameters[par_name]

    # Set the mock behavior
    array_model.site_model.get_parameter_value.side_effect = mock_get_parameter_value

    with (
        mock.patch("simtools.corsika.corsika_config.ModelParameter") as mp,
        mock.patch.object(
            settings._Config,
            "args",
            new_callable=mock.PropertyMock,
            return_value=corsika_config_data,
        ),
    ):
        mp_instance = mp.return_value
        mp_instance.get_simulation_software_parameters.return_value = {
            "corsika_iact_max_bunches": {"value": 1000000, "unit": None},
            "corsika_cherenkov_photon_bunch_size": {"value": 5.0, "unit": None},
            "corsika_cherenkov_photon_wavelength_range": {
                "value": [240.0, 1000.0],
                "unit": "nm",
            },
            "corsika_first_interaction_height": {"value": 0.0, "unit": "cm"},
            "corsika_particle_kinetic_energy_cutoff": {
                "value": [0.3, 0.1, 0.020, 0.020],
                "unit": "GeV",
            },
            "corsika_longitudinal_shower_development": {"value": 20.0, "unit": "g/cm2"},
            "corsika_iact_split_auto": {"value": 15000000, "unit": None},
            "corsika_starting_grammage": {"value": 0.0, "unit": "g/cm2"},
            "corsika_iact_io_buffer": {"value": 800, "unit": "MB"},
        }

        corsika_config = CorsikaConfig(
            array_model=array_model, run_number=1, label="test-corsika-config"
        )

    corsika_config.array_model.site = "South"
    return corsika_config


@pytest.fixture
def corsika_runner_mock_array_model(corsika_config_mock_array_model, io_handler):
    return CorsikaRunner(
        corsika_config=corsika_config_mock_array_model,
        label="test-corsika-runner",
        use_multipipe=False,
    )


@pytest.fixture
def file_has_text():
    """Check if a file contains a specific text."""

    def wrapper(file, text):
        try:
            with (
                open(file, "rb", 0) as string_file,
                mmap.mmap(string_file.fileno(), 0, access=mmap.ACCESS_READ) as text_file_input,
            ):
                re_search_1 = re.compile(f"{text}".encode())
                search_result_1 = re_search_1.search(text_file_input)
                if search_result_1 is None:
                    return False

                return True
        except FileNotFoundError:
            return False
        except ValueError:
            return False

    return wrapper


@pytest.fixture
def safe_tar_open():
    """Fixture returning a context manager to open tar files safely in tests.

    Archives are created/read from controlled test inputs; no extraction to filesystem occurs.
    """

    @contextmanager
    def _open(path, mode):
        with tarfile.open(path, mode) as tar:  # NOSONAR
            yield tar

    return _open


@pytest.fixture
def corsika_file_gamma():
    """Gamma corsika file for testing."""
    return (
        "tests/resources/"
        "gamma_run000007_za40deg_azm180deg_South_subsystem_lsts_6.0.2_test.corsika.zst"
    )


@pytest.fixture
def sim_telarray_file_gamma():
    """Gamma sim_telarray file for testing."""
    return (
        "tests/resources/"
        "gamma_diffuse_run000010_za20deg_azm000deg_North_alpha_6.0.0_test_file.simtel.zst"
    )


@pytest.fixture
def sim_telarray_file_proton():
    """Proton sim_telarray file for testing."""
    return (
        "tests/resources/proton_run000201_za20deg_azm000deg_North_alpha_6.0.0_test_file.simtel.zst"
    )


@pytest.fixture
def sim_telarray_hdata_file_gamma():
    """Gamma sim_telarray histogram file for testing."""
    return "tests/resources/gamma_run2_za20deg_azm0deg-North-Prod5_test-production-5.hdata.zst"

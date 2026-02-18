"""Common fixtures for unit tests."""

import functools
import json
import logging
import mmap
import os
import re
import tarfile
from contextlib import ExitStack, contextmanager
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from unittest import mock
from unittest.mock import MagicMock, PropertyMock, patch

import matplotlib.pyplot as plt
import pytest
from astropy import units as u
from dotenv import load_dotenv

import simtools.io.io_handler
from simtools import settings
from simtools.configuration.configurator import Configurator
from simtools.corsika.corsika_config import CorsikaConfig
from simtools.db import db_handler
from simtools.db.mongo_db import MongoDBHandler
from simtools.model.array_model import ArrayModel
from simtools.model.site_model import SiteModel
from simtools.model.telescope_model import TelescopeModel
from simtools.runners.corsika_runner import CorsikaRunner

logger = logging.getLogger()

UNIT_TEST_DB = "unit_tests/db"


@functools.lru_cache
def _load_mock_db_json(file_name):
    mock_db_dir = Path(__file__).resolve().parent.parent / "resources" / "mock_db"
    file_path = mock_db_dir / file_name
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _apply_mock_param_defaults(parameters):
    return {
        name: {
            **param,
            "parameter_version": param.get("parameter_version", "1.0.0"),
            "model_parameter_schema_version": param.get("model_parameter_schema_version", "1.0.0"),
        }
        for name, param in parameters.items()
    }


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


# Array element configuration for mock database
_ARRAY_ELEMENT_COUNTS = {"LSTN": 4, "LSTS": 4, "MSTN": 5, "MSTS": 11, "SSTS": 5}
_ARRAY_ELEMENT_TYPES = list(_ARRAY_ELEMENT_COUNTS.keys())
_TELESCOPE_TYPE_TO_DESIGN_MODEL = {"LST": "LSTN-design", "MST": "MSTN-design", "SST": "SSTS-design"}


def _format_elements(prefix, count=None):
    """Format array element names (e.g., 'LSTN-01', 'LSTN-02')."""
    if count is None:
        count = _ARRAY_ELEMENT_COUNTS[prefix]
    return [f"{prefix}-{idx:02d}" for idx in range(1, count + 1)]


def _format_all_array_elements():
    """Format all available array elements."""
    return list(
        chain.from_iterable(_format_elements(elem_type) for elem_type in _ARRAY_ELEMENT_TYPES)
    )


def _get_design_model_for_element(array_element_name):
    """Get design model for given array element."""
    if not array_element_name or array_element_name.endswith("-design"):
        return array_element_name
    for prefix, design_model in _TELESCOPE_TYPE_TO_DESIGN_MODEL.items():
        if prefix in array_element_name:
            return design_model
    return None


def _get_site_params(site, site_specific_params_north, site_specific_params_south):
    """Get site-specific parameters based on site name."""
    return site_specific_params_north if site == "North" else site_specific_params_south


def _mock_get_model_parameters_impl(
    site,
    array_element_name,
    collection,
    model_version,
    mock_parameters,
    site_specific_params_north,
    site_specific_params_south,
):
    """Implementation of mock_get_model_parameters."""
    params = dict(mock_parameters)

    # Add SSTS-specific parameters if needed
    if array_element_name and "SSTS" in array_element_name:
        params.update(
            {
                "effective_focal_length": {
                    "value": 2.15191,
                    "parameter_version": "1.0.0",
                    "type": "float64",
                    "unit": "m",
                    "file": False,
                    "model_parameter_schema_version": "1.0.0",
                },
                "mirror_focal_length": {
                    "value": 2.15,
                    "parameter_version": "1.0.0",
                    "type": "float64",
                    "unit": "m",
                    "file": False,
                    "model_parameter_schema_version": "1.0.0",
                },
            }
        )

    # Apply site-specific overrides
    site_params = _get_site_params(site, site_specific_params_north, site_specific_params_south)
    params.update(site_params)
    return params


def _mock_get_model_parameter_impl(
    parameter,
    site,
    array_element_name,
    parameter_version,
    model_version,
    mock_parameters,
    site_specific_params_north,
    site_specific_params_south,
):
    """Implementation of mock_get_model_parameter."""
    if model_version is not None and isinstance(model_version, list):
        raise ValueError("Only one model version can be passed to get_model_parameter, not a list.")

    params = dict(mock_parameters)
    site_params = _get_site_params(site, site_specific_params_north, site_specific_params_south)
    params.update(site_params)

    if parameter in params:
        param_data = params[parameter]
        if parameter_version is None or param_data.get("parameter_version") == parameter_version:
            return param_data

    raise ValueError(f"Parameter {parameter} with version {parameter_version} not found")


def _mock_export_model_files(*args, **kwargs):
    """Mock export_model_files (no file I/O)."""
    return {}


def _mock_get_ecsv_file_as_astropy_table(*args, **kwargs):
    """Mock get_ecsv_file_as_astropy_table with Quantity columns."""
    from astropy.table import Column, Table

    table = Table()
    table["wavelength"] = Column([300.0, 400.0, 500.0, 600.0, 700.0] * u.nm)
    table["differential photon rate"] = Column(
        [1.0, 1.2, 1.0, 0.8, 0.5] / (u.nm * u.cm**2 * u.ns * u.sr)
    )
    return table


def _mock_get_array_elements_of_type(array_element_type, all_elements):
    """Filter array elements by type prefix."""
    return [elem for elem in all_elements if elem.startswith(array_element_type)]


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
def mock_db_handler(request):
    """
    Mock DatabaseHandler for unit tests.

    Provides common mock behaviors to avoid real database connections.
    Returns a MagicMock configured with typical DatabaseHandler methods.
    Tests in tests/unit_tests/db/ receive a real DatabaseHandler instance.
    """
    test_file_path = str(request.node.fspath)
    if UNIT_TEST_DB in test_file_path:
        db_instance = request.getfixturevalue("db")
        db_instance.get_model_versions = MagicMock(return_value=["1.0.0", "5.0.0", "6.0.0"])
        return db_instance

    # Load mock data from JSON files
    mock_parameters = _apply_mock_param_defaults(_load_mock_db_json("mock_parameters.json"))
    mock_sim_config_params = _apply_mock_param_defaults(
        _load_mock_db_json("mock_sim_config_params.json")
    )
    site_specific_params_north = _apply_mock_param_defaults(
        _load_mock_db_json("site_params_north.json")
    )
    site_specific_params_south = _apply_mock_param_defaults(
        _load_mock_db_json("site_params_south.json")
    )

    # Create closures for mock functions with captured data
    def mock_get_model_parameters(site, array_element_name, collection, model_version, **kwargs):
        return _mock_get_model_parameters_impl(
            site,
            array_element_name,
            collection,
            model_version,
            mock_parameters,
            site_specific_params_north,
            site_specific_params_south,
        )

    def mock_get_model_parameter(
        parameter, site, array_element_name, parameter_version=None, model_version=None, **kwargs
    ):
        return _mock_get_model_parameter_impl(
            parameter,
            site,
            array_element_name,
            parameter_version,
            model_version,
            mock_parameters,
            site_specific_params_north,
            site_specific_params_south,
        )

    def mock_get_design_model(
        model_version, array_element_name=None, collection="telescopes", **kwargs
    ):
        return _get_design_model_for_element(array_element_name)

    # Pre-format all array elements
    all_array_elements = _format_all_array_elements()

    def mock_get_array_elements_of_type(
        array_element_type, model_version=None, collection=None, **kwargs
    ):
        return _mock_get_array_elements_of_type(array_element_type, all_array_elements)

    # Configure mock database
    mock_db = MagicMock()
    mock_db.is_configured.return_value = True
    mock_db.get_design_model.side_effect = mock_get_design_model
    mock_db.get_model_parameters.side_effect = mock_get_model_parameters
    mock_db.get_model_parameter.side_effect = mock_get_model_parameter
    mock_db.get_model_parameters_for_all_model_versions.return_value = {}
    mock_db.get_model_versions.return_value = ["6.0.2", "5.0.0"]
    mock_db.get_array_elements.return_value = _format_elements("LSTN", 1) + list(
        chain.from_iterable(_format_elements(t) for t in ["LSTS", "MSTN", "MSTS", "SSTS"])
    )
    mock_db.get_simulation_configuration_parameters.return_value = mock_sim_config_params
    mock_db.get_array_elements_of_type.side_effect = mock_get_array_elements_of_type
    mock_db.export_model_files.side_effect = _mock_export_model_files
    mock_db.export_model_file.return_value = None
    mock_db.get_ecsv_file_as_astropy_table.side_effect = _mock_get_ecsv_file_as_astropy_table
    mock_db.db_name = "test_db"

    return mock_db


@pytest.fixture(autouse=True)
def mock_database_handler(request, mocker):
    """
    Automatically mock DatabaseHandler for all unit tests except those in db/.

    This prevents unit tests from trying to connect to real databases.
    Tests in tests/unit_tests/db/ are excluded from this mocking.
    Also mocks model parameter schema validation to avoid schema version mismatches.
    """
    test_file_path = str(request.node.fspath)

    # Skip mocking for tests in db/ directory
    if UNIT_TEST_DB in test_file_path:
        yield
        return

    mock_db_handler = request.getfixturevalue("mock_db_handler")

    # Mock schema validation to avoid version check issues
    mocker.patch("simtools.model.model_parameter.ModelParameter._check_model_parameter_versions")

    # Mock DatabaseHandler for all other unit tests
    with mock.patch("simtools.db.db_handler.DatabaseHandler", return_value=mock_db_handler):
        yield


@pytest.fixture
def db(request):
    """Database object with configuration from settings.config.db_handler."""

    test_file_path = str(request.node.fspath)
    if UNIT_TEST_DB not in test_file_path:
        db_instance = db_handler.DatabaseHandler()
        yield db_instance
        return

    request.getfixturevalue("reset_db_client")

    db_config = {
        "db_server": "localhost",
        "db_api_port": 27017,
        "db_api_user": "user",
        "db_api_pw": "pw",
        "db_api_authentication_database": "admin",
        "db_simulation_model": "CTAO-Simulation-Model",
        "db_simulation_model_version": "v0-12-0",
    }
    previous_state = {
        "_args": settings.config._args,
        "_db_config": settings.config._db_config,
        "_sim_telarray_path": settings.config._sim_telarray_path,
        "_sim_telarray_exe": settings.config._sim_telarray_exe,
        "_corsika_path": settings.config._corsika_path,
        "_corsika_interaction_table_path": settings.config._corsika_interaction_table_path,
        "_corsika_exe": settings.config._corsika_exe,
    }
    previous_db_client = MongoDBHandler.db_client

    settings.config._args = MappingProxyType({"corsika_path": None, "sim_telarray_path": None})
    settings.config._db_config = MappingProxyType(db_config)
    settings.config._sim_telarray_path = None
    settings.config._sim_telarray_exe = None
    settings.config._corsika_path = None
    settings.config._corsika_interaction_table_path = None
    settings.config._corsika_exe = None

    # Create a mock MongoClient that properly handles close()
    mock_mongo_client = MagicMock()
    mock_mongo_client.close = MagicMock()

    with patch("simtools.db.mongo_db.MongoClient", return_value=mock_mongo_client):
        db_instance = db_handler.DatabaseHandler()
        MongoDBHandler.db_client = MongoDBHandler.db_client or mock_mongo_client
        yield db_instance
        # Explicitly close the mock client to avoid unraisable exception warnings
        if hasattr(MongoDBHandler.db_client, "close"):
            try:
                MongoDBHandler.db_client.close()
            except Exception as exc:
                # Ignore close errors in tests to avoid masking real test failures
                logger.debug("Ignoring exception while closing mock MongoDB client: %r", exc)

    settings.config._args = previous_state["_args"]
    settings.config._db_config = previous_state["_db_config"]
    settings.config._sim_telarray_path = previous_state["_sim_telarray_path"]
    settings.config._sim_telarray_exe = previous_state["_sim_telarray_exe"]
    settings.config._corsika_path = previous_state["_corsika_path"]
    settings.config._corsika_interaction_table_path = previous_state[
        "_corsika_interaction_table_path"
    ]
    settings.config._corsika_exe = previous_state["_corsika_exe"]
    MongoDBHandler.db_client = previous_db_client


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


# Test data file utilities
_TEST_DATA_FILES = {
    (
        "corsika",
        "gamma",
    ): "tests/resources/gamma_run000007_za40deg_azm180deg_South_subsystem_lsts_6.0.2_test.corsika.zst",
    (
        "sim_telarray",
        "gamma",
    ): "tests/resources/gamma_diffuse_run000010_za20deg_azm000deg_North_alpha_6.0.0_test_file.simtel.zst",
    (
        "sim_telarray",
        "proton",
    ): "tests/resources/proton_run000201_za20deg_azm000deg_North_alpha_6.0.0_test_file.simtel.zst",
    (
        "sim_telarray_hdata",
        "gamma",
    ): "tests/resources/gamma_run2_za20deg_azm0deg-North-Prod5_test-production-5.hdata.zst",
    ("telescope_positions", "North"): "tests/resources/telescope_positions-North-ground.ecsv",
    (
        "telescope_positions",
        "North-calibration",
    ): "tests/resources/telescope_positions-North-with-calibration-devices-ground.ecsv",
    ("telescope_positions", "North-utm"): "tests/resources/telescope_positions-North-utm.ecsv",
    ("telescope_positions", "South"): "tests/resources/telescope_positions-South-ground.ecsv",
}


@pytest.fixture
def get_test_data_file():
    """Fixture providing test data file path retrieval.

    Returns
    -------
    callable
        Function to get test data file path by file type and variant.
        Call as: get_test_data_file(file_type, variant="gamma")

    Examples
    --------
    >>> corsika_path = get_test_data_file("corsika", "gamma")
    >>> pos_path = get_test_data_file("telescope_positions", "North")
    """

    def _get_test_data_file(file_type, variant="gamma"):
        """Get test data file path by file type and variant.

        Parameters
        ----------
        file_type : str
            Type of file: "corsika", "sim_telarray", "sim_telarray_hdata", "telescope_positions"
        variant : str, optional
            Variant of file: "gamma" (default for simulation files), "proton", "North", "South", "utm", etc.

        Returns
        -------
        str
            Path to test data file

        Raises
        ------
        KeyError
            If the requested file type and variant combination is not available
        """
        key = (file_type, variant)
        if key not in _TEST_DATA_FILES:
            available = ", ".join(f"{ft}[{v}]" for ft, v in _TEST_DATA_FILES.keys())
            raise KeyError(
                f"Test data file not found for {file_type}[{variant}]. Available: {available}"
            )
        return _TEST_DATA_FILES[key]

    return _get_test_data_file

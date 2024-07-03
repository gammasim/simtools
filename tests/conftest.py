import logging
import mmap
import os
import re
from pathlib import Path
from unittest import mock

import pytest
from astropy import units as u
from dotenv import dotenv_values, load_dotenv

import simtools.io_operations.io_handler
from simtools.configuration.configurator import Configurator
from simtools.db import db_handler
from simtools.model.array_model import ArrayModel
from simtools.model.site_model import SiteModel
from simtools.model.telescope_model import TelescopeModel

logger = logging.getLogger()


@pytest.fixture()
def tmp_test_directory(tmpdir_factory):
    """Sets temporary test directories. Some tests depend on this structure."""

    tmp_test_dir = tmpdir_factory.mktemp("test-data")
    tmp_sub_dirs = ["resources", "output", "simtel", "model", "application-plots"]
    for sub_dir in tmp_sub_dirs:
        tmp_sub_dir = tmp_test_dir / sub_dir
        tmp_sub_dir.mkdir()

    return tmp_test_dir


@pytest.fixture()
def io_handler(tmp_test_directory):
    """Define io_handler fixture including output and model directories."""
    tmp_io_handler = simtools.io_operations.io_handler.IOHandler()
    tmp_io_handler.set_paths(
        output_path=str(tmp_test_directory) + "/output",
        data_path="./data/",
        model_path=str(tmp_test_directory) + "/model",
    )
    return tmp_io_handler


@pytest.fixture()
def _mock_settings_env_vars(tmp_test_directory):
    """Removes all environment variable from the test system and explicitly sets those needed."""
    _url = (
        "https://gitlab.cta-observatory.org/cta-science/simulations/"
        "simulation-model/model_parameters/-/raw/main"
    )

    with mock.patch.dict(
        os.environ,
        {
            "SIMTOOLS_SIMTEL_PATH": str(tmp_test_directory) + "/simtel",
            "SIMTOOLS_DB_API_USER": "db_user",
            "SIMTOOLS_DB_API_PW": "12345",
            "SIMTOOLS_DB_API_PORT": "42",
            "SIMTOOLS_DB_SERVER": "abc@def.de",
            "SIMTOOLS_DB_SIMULATION_MODEL": "sim_model",
            "SIMTOOLS_DB_SIMULATION_MODEL_URL": _url,
        },
        clear=True,
    ):
        yield


@pytest.fixture()
def simtel_path():
    """Empty string used as placeholder for simtel_path."""
    return Path("")


@pytest.fixture()
def simtel_path_no_mock():
    """Simtel path as set by the .env file."""
    load_dotenv(".env")
    simtel_path = Path(os.path.expandvars("$SIMTOOLS_SIMTEL_PATH"))
    if simtel_path.exists():
        return simtel_path
    return ""


@pytest.fixture()
def args_dict(tmp_test_directory, simtel_path):
    """Minimal configuration from command line."""
    return Configurator().default_config(
        (
            "--output_path",
            str(tmp_test_directory),
            "--data_path",
            "./data/",
            "--simtel_path",
            str(simtel_path),
        ),
    )


@pytest.fixture()
def args_dict_site(tmp_test_directory, simtel_path):
    "Configuration include site and telescopes."
    return Configurator().default_config(
        (
            "--output_path",
            str(tmp_test_directory),
            "--data_path",
            "./data/",
            "--simtel_path",
            str(simtel_path),
            "--site",
            "South",
            "--telescope",
            "MSTS-07",
            "--label",
            "integration_test",
        )
    )


@pytest.fixture()
def db_config():
    """DB configuration from .env file."""

    mongo_db_config = {
        key.lower().replace("simtools_", ""): value
        for key, value in dict(dotenv_values(".env")).items()
    }
    _db_para = (
        "db_api_user",
        "db_api_pw",
        "db_api_port",
        "db_server",
        "db_simulation_model",
        "db_simulation_model_url",
    )
    for _para in _db_para:
        if _para not in mongo_db_config:
            mongo_db_config[_para] = os.environ.get(f"SIMTOOLS_{_para.upper()}")
    if mongo_db_config["db_api_port"] is not None:
        mongo_db_config["db_api_port"] = int(mongo_db_config["db_api_port"])
    return mongo_db_config


@pytest.fixture()
def simulation_model_url(db_config):
    """Simulation model URL from .env file or default."""
    if (
        db_config["db_simulation_model_url"] is None
        or len(db_config["db_simulation_model_url"]) == 0
    ):
        db_config["db_simulation_model_url"] = (
            "https://gitlab.cta-observatory.org/cta-science/simulations/"
            "simulation-model/model_parameters/-/raw/main/"
        )
    return db_config["db_simulation_model_url"]


@pytest.fixture()
def db(db_config):
    """Database object with configuration from .env file."""
    return db_handler.DatabaseHandler(mongo_db_config=db_config)


def pytest_addoption(parser):
    """Model version command line parameter."""
    parser.addoption("--model_version", action="store", default=None)


@pytest.fixture()
def model_version():
    """Simulation model version used in tests."""
    return "2024-02-01"


@pytest.fixture()
def array_model_north(simulator_config_data_north, io_handler, db_config, model_version):
    """Array model for North site."""
    return ArrayModel(
        label="test-lst-array",
        site=simulator_config_data_north["common"]["site"],
        layout_name=simulator_config_data_north["common"]["layout_name"],
        mongo_db_config=db_config,
        model_version=model_version,
    )


@pytest.fixture()
def array_model_south(simulator_config_data_south, io_handler, db_config, model_version):
    """Array model for South site."""
    return ArrayModel(
        label="test-lst-array",
        site=simulator_config_data_south["common"]["site"],
        layout_name=simulator_config_data_south["common"]["layout_name"],
        mongo_db_config=db_config,
        model_version=model_version,
    )


@pytest.fixture()
def site_model_south(db_config, model_version):
    """Site model for South site."""
    return SiteModel(
        site="South",
        mongo_db_config=db_config,
        label="site-south",
        model_version=model_version,
    )


@pytest.fixture()
def site_model_north(db_config, model_version):
    """Site model for North site."""
    return SiteModel(
        site="North",
        mongo_db_config=db_config,
        label="site-north",
        model_version=model_version,
    )


@pytest.fixture()
def telescope_model_lst(db_config, io_handler, model_version):
    """Telescope model LST North."""
    return TelescopeModel(
        site="North",
        telescope_name="LSTN-01",
        model_version=model_version,
        mongo_db_config=db_config,
        label="test-telescope-model-lst",
    )


@pytest.fixture()
def telescope_model_mst(db_config, io_handler, model_version):
    """Telescope model MST South."""
    return TelescopeModel(
        site="South",
        telescope_name="MSTS-design",
        model_version=model_version,
        label="test-telescope-model-mst",
        mongo_db_config=db_config,
    )


@pytest.fixture()
def telescope_model_sst(db_config, io_handler, model_version):
    """Telescope model SST South."""
    return TelescopeModel(
        site="South",
        telescope_name="SSTS-design",
        model_version=model_version,
        mongo_db_config=db_config,
        label="test-telescope-model-sst",
    )


# TODO - keep prod5 until a complete prod6 model is in the DB
@pytest.fixture()
def telescope_model_sst_prod5(db_config, io_handler):
    """Telescope model SST South (prod5)."""
    return TelescopeModel(
        site="South",
        telescope_name="SSTS-design",
        model_version="Prod5",
        mongo_db_config=db_config,
        label="test-telescope-model-sst",
    )


@pytest.fixture()
def telescope_north_with_calibration_devices_test_file():
    """Telescope positions North with calibration devices."""
    return "tests/resources/telescope_positions-North-with-calibration-devices-ground.ecsv"


@pytest.fixture()
def telescope_north_test_file():
    """Telescope positions North."""
    return "tests/resources/telescope_positions-North-ground.ecsv"


@pytest.fixture()
def telescope_north_utm_test_file():
    """Telescope positions North (UTM coordinates)."""
    return "tests/resources/telescope_positions-North-utm.ecsv"


@pytest.fixture()
def telescope_south_test_file():
    """Telescope positions South."""
    return "tests/resources/telescope_positions-South-ground.ecsv"


@pytest.fixture()
def corsika_output_file_name():
    """CORSIKA output file name for testing."""
    return "tests/resources/tel_output_10GeV-2-gamma-20deg-CTAO-South.corsikaio"


@pytest.fixture()
def corsika_histograms_instance(io_handler, corsika_output_file_name):
    """Corsika histogram instance."""
    from simtools.corsika.corsika_histograms import CorsikaHistograms

    return CorsikaHistograms(
        corsika_output_file_name, output_path=io_handler.get_output_directory(dir_type="test")
    )


@pytest.fixture()
def corsika_histograms_instance_set_histograms(db, io_handler, corsika_histograms_instance):
    """Corsika histogram instance (fully configured)."""
    corsika_histograms_instance.set_histograms()
    return corsika_histograms_instance


@pytest.fixture()
def simulator_config_data_north(tmp_test_directory):
    return {
        "common": {
            "site": "North",
            "layout_name": "test_layout",
            "data_directory": f"{tmp_test_directory!s}/test-output",
            "zenith": 20 * u.deg,
            "azimuth": 0 * u.deg,
            "primary": "gamma",
        },
        "showers": {
            "eslope": -2.5,
            "viewcone": [0 * u.deg, 0 * u.deg],
            "nshow": 10,
            "erange": [100 * u.GeV, 1 * u.TeV],
            "cscat": [10, 1400 * u.m, 0],
            "run_list": [3, 4],
            "run_range": [6, 10],
        },
        "array": {},
    }


@pytest.fixture()
def array_config_data(simulator_config_data):
    return simulator_config_data["common"] | simulator_config_data["array"]


@pytest.fixture()
def shower_config_data_north(simulator_config_data_north):
    return simulator_config_data_north["common"] | simulator_config_data_north["showers"]


@pytest.fixture()
def simulator_config_data_south(tmp_test_directory):
    return {
        "common": {
            "site": "South",
            "layout_name": "test_layout",
            "data_directory": f"{tmp_test_directory!s}/test-output",
            "zenith": 20 * u.deg,
            "azimuth": 0 * u.deg,
            "primary": "gamma",
        },
        "showers": {
            "eslope": -2.5,
            "viewcone": [0 * u.deg, 0 * u.deg],
            "nshow": 10,
            "erange": [100 * u.GeV, 1 * u.TeV],
            "cscat": [10, 1400 * u.m, 0],
            "run_list": [3, 4],
            "run_range": [6, 10],
        },
        "array": {},
    }


@pytest.fixture()
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

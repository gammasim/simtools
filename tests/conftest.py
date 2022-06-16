import os
import logging
from pathlib import Path
import pytest
import yaml

import simtools.config as cfg

logger = logging.getLogger()


@pytest.fixture
def cfg_setup(tmp_configuration_test_file):
    cfg.setConfigFileName(str(tmp_configuration_test_file))


@pytest.fixture
def tmp_test_directory(tmpdir_factory):
    tmp_test_dir = tmpdir_factory.mktemp("test-data")
    tmp_sub_dirs = ['/resources/', '/output/', '/simtel/']
    for sub_dir in tmp_sub_dirs:
        tmp_sub_dir = tmp_test_dir + sub_dir
        tmp_sub_dir.mkdir()

    return tmp_test_dir


@pytest.fixture
def configuration_parameters(tmp_test_directory):

    return {
        'modelFilesLocations':  [
            str(tmp_test_directory) + '/resources/',
            './tests/resources/'
        ],
        'dataLocation': './data/',
        'outputLocation': str(tmp_test_directory) + '/output/',
        'simtelPath': str(tmp_test_directory) + '/simtel/',
        'useMongoDB': False,
        'mongoDBConfigFile': None,
        'extraCommands': [''],
    }


@pytest.fixture
def tmp_configuration_test_file(tmp_test_directory,
                                configuration_parameters):
    tmp_test_dir = tmp_test_directory
    tmp_test_file = tmp_test_dir / 'config-test.yml'
    tmp_test_config = dict(configuration_parameters)

    with open(tmp_test_file, "w") as output:
        yaml.safe_dump(
            tmp_test_config,
            output,
            sort_keys=False)
    return str(tmp_test_file)


@pytest.fixture
def db_connection(tmp_test_directory):
    # prefer local dbDetails.yml file for testing
    try:
        dbDetailsFile = "dbDetails.yml"
        with open(dbDetailsFile, "r") as stream:
            yaml.load(stream, Loader=yaml.FullLoader)
        return dbDetailsFile
    # try if DB details are defined in environment
    # (e.g., as secrets in github actions)
    except FileNotFoundError:
        parsToDbDetails = {}
        environVariblesToCheck = {
            "mongodbServer": "DB_API_NAME",
            "userDB": "DB_API_USER",
            "passDB": "DB_API_PW",
            "dbPort": "DB_API_PORT",
        }
        found_env = True
        for par, env in environVariblesToCheck.items():
            if env in os.environ:
                parsToDbDetails[par] = os.environ[env]
            else:
                found_env = False
        if found_env:
            dbDetailsFileName = str(tmp_test_directory) + "dbDetails.yml"
            cfg.createDummyDbDetails(filename=dbDetailsFileName, **parsToDbDetails)
            return dbDetailsFileName

        return ""


@pytest.fixture
def set_db(cfg_setup, db_connection):

    if len(str(db_connection)) == 0:
        pytest.skip("Test requires database (DB) connection")

    cfg.change('useMongoDB', True)
    cfg.change('mongoDBConfigFile', str(db_connection))


@pytest.fixture
def set_simtelarray(cfg_setup):

    if 'SIMTELPATH' in os.environ:
        simtelpath = Path(os.path.expandvars('$SIMTELPATH'))
        if simtelpath.exists():
            cfg.change('simtelPath', str(simtelpath))
        else:
            pytest.skip('sim_telarray not found in {}'.format(simtelpath))
    else:
        pytest.skip('SIMTELPATH')

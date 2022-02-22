import os


CONFIG_FILE_MSG = "The simtools configuration file (config.yml) is not available"

DB_CONNECTION_MSG = "Connection with the DB is not available"

SIMTEL_MSG = "sim_telarray installation is not available"


def _collect_conftest_flag(name):
    flag = os.environ.get(name, "0")
    return (flag == "1")


def has_config_file():
    return _collect_conftest_flag("HAS_CONFIG_FILE")


def has_db_connection():
    return _collect_conftest_flag("HAS_DB_CONNECTION")


def simtel_installed():
    return _collect_conftest_flag("SIMTEL_INSTALLED")

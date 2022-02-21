import os


CONFIG_FILE_MSG = "Configuration file is not available"

DB_CONNECTION_MSG = "There is not connection with DB"

SIMTEL_MSG = "sim_telarray is not installed"


def _collect_conftest_flag(name):
    flag = os.environ.get(name, "0")
    return (flag == "1")


def has_config_file():
    return _collect_conftest_flag("HAS_CONFIG_FILE")


def has_db_connection():
    return _collect_conftest_flag("HAS_DB_CONNECTION")


def simtel_installed():
    return _collect_conftest_flag("SIMTEL_INSTALLED")

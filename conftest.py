import os
import logging

import simtools.config as cfg
import simtools.db_handler as db

logger = logging.getLogger()


try:
    cfg.loadConfig()
except FileNotFoundError:
    logger.debug("simtools configuration file was NOT found")
    logger.debug("Setting HAS_CONFIG_FILE = 0")
    logger.debug("Setting SIMTEL_INSTALLED = 0")
    logger.debug("Setting HAS_DB_CONNECTION = 0")
    os.environ["HAS_CONFIG_FILE"] = "0"
    os.environ["SIMTEL_INSTALLED"] = "0"
    os.environ["HAS_DB_CONNECTION"] = "0"
else:
    os.environ["HAS_CONFIG_FILE"] = "1"
    logger.debug("simtools configuration found WAS found")
    logger.debug("Setting HAS_CONFIG_FILE = 1")

    # Checking whether sim_telarray is properly installed
    simtelPath = cfg.get("simtelPath")
    simtelBinPath = simtelPath + "/sim_telarray/bin/sim_telarray"
    os.environ["SIMTEL_INSTALLED"] = "1" if os.path.exists(simtelBinPath) else "0"
    logger.debug("Setting SIMTEL_INSTALLED = {}".format(os.environ["SIMTEL_INSTALLED"]))

    # Checking whether there is DB connection
    if not cfg.get("useMongoDB"):
        os.environ["HAS_DB_CONNECTION"] = "0"
    else:
        # Trying to connect to the DB
        db.DatabaseHandler()
        os.environ["HAS_DB_CONNECTION"] = "1"
        print('DB CONNECTEDDDDDDDDDDDDDD')

    logger.debug("Setting HAS_DB_CONNECTION = {}".format(os.environ["HAS_DB_CONNECTION"]))


def pytest_sessionfinish(session, exitstatus):
    """ Cleaning up output files before ending the pytest session. """
    os.system('./clean_files')

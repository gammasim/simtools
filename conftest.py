import os
import logging

import simtools.config as cfg
from simtools import db_handler

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

    # Creating a dummy config.yml file
    configFilename = 'configDummy.yml'
    cfg.createDummyConfigFile(configFilename)
    cfg.setConfigFileName(configFilename)

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
        db = db_handler.DatabaseHandler()
        try:
            db.getModelParameters("north", "lst-1", "Current")
            os.environ["HAS_DB_CONNECTION"] = "1"
            logger.debug("DB connection is available")
            logger.debug("Setting HAS_DB_CONNECTION = 1")
        except Exception:
            os.environ["HAS_DB_CONNECTION"] = "0"
            logger.debug("DB connection is NOT available")
            logger.debug("Setting HAS_DB_CONNECTION = 0")


def pytest_sessionfinish(session, exitstatus):
    """ Cleaning up output files before ending the pytest session. """
    os.system("./clean_files")
    os.system("rm configDummy.yml | true")

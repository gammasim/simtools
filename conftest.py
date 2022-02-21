import os

import simtools.config as cfg


try:
    cfg.loadConfig()
except FileNotFoundError:
    os.environ["HAS_CONFIG_FILE"] = "0"
    os.environ["SIMTEL_INSTALLED"] = "0"
    os.environ["HAS_DB_CONNECTION"] = "0"
else:

    os.environ["HAS_CONFIG_FILE"] = "1"

    # Checking whether sim_telarray is properly installed
    simtelPath = cfg.get("simtelPath")
    simtelBinPath = simtelPath + "/sim_telarray/bin/sim_telarray"
    os.environ["SIMTEL_INSTALLED"] = "1" if os.path.exists(simtelBinPath) else "0"

    # Checking whether there is DB connection
    useMongoDB = cfg.get("useMongoDB")
    os.environ["HAS_DB_CONNECTION"] = "1" if useMongoDB else "0"

#!/usr/bin/python3

import pytest
import logging
import matplotlib.pyplot as plt

import simtools.io_handler as io
from simtools.model.telescope_model import TelescopeModel
from simtools.camera_efficiency import CameraEfficiency
from simtools.util.tests import (
    has_db_connection,
    simtel_installed,
    DB_CONNECTION_MSG,
    SIMTEL_MSG,
)


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.mark.skipif(not has_db_connection(), reason=DB_CONNECTION_MSG)
@pytest.mark.skipif(not simtel_installed(), reason=SIMTEL_MSG)
def test_main():
    label = "test_camera_eff"
    tel = TelescopeModel(
        site="North",
        telescopeModelName="LST-1",
        label=label
    )

    ce = CameraEfficiency(telescopeModel=tel)
    ce.simulate(force=True)
    ce.analyze(force=True)

    # Plotting Cherenkov
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel("wl")
    ax.set_ylabel("eff")

    ce.plot("cherenkov")

    plotFileCherenkov = io.getTestPlotFile("camera_eff_cherenkov.pdf")
    plt.savefig(plotFileCherenkov)

    # Plotting NSB
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_yscale("log")
    ax.set_xlabel("wl")
    ax.set_ylabel("eff")

    ce.plot("nsb")

    plotFileNSB = io.getTestPlotFile("camera_eff_nsb.pdf")
    plt.savefig(plotFileNSB)


if __name__ == "__main__":

    test_main()

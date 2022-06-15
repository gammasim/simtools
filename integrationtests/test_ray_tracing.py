#!/usr/bin/python3

import pytest
import logging
import matplotlib.pyplot as plt
from copy import copy

import astropy.units as u

import simtools.io_handler as io
from simtools.ray_tracing import RayTracing
from simtools.model.telescope_model import TelescopeModel
from simtools.util.tests import (
    has_db_connection,
    simtel_installed,
    SIMTEL_MSG,
    DB_CONNECTION_MSG,
)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.mark.skipif(not simtel_installed(), reason=SIMTEL_MSG)
def test_run_no_db():
    cfgFile = io.getTestDataFile("CTA-North-LST-1-Current_test-telescope-model.cfg")
    configData = {
        "sourceDistance": 10 * u.km,
        "zenithAngle": 20 * u.deg,
        "offAxisAngle": [0, 1.0, 2.0, 3.0, 4.0] * u.deg,
    }
    tel = TelescopeModel.fromConfigFile(
        site="North",
        telescopeModelName="LST-1",
        label="test-run-no-db",
        configFileName=cfgFile,
    )

    ray = RayTracing(telescopeModel=tel, configData=configData)
    ray.simulate(test=True, force=True)
    ray.analyze(force=True)


@pytest.mark.skipif(not has_db_connection(), reason=DB_CONNECTION_MSG)
@pytest.mark.skipif(not simtel_installed(), reason=SIMTEL_MSG)
@pytest.mark.parametrize("telescopeModelName", ["sst-1M", "sst-ASTRI", "sst-GCT"])
def test_ssts(telescopeModelName):
    # Test with 3 SSTs
    version = "prod3"
    configData = {
        "sourceDistance": 10 * u.km,
        "zenithAngle": 20 * u.deg,
        "offAxisAngle": [0, 1.0, 2.0, 3.0, 4.0] * u.deg,
    }
    tel = TelescopeModel(
        site="south",
        telescopeModelName=telescopeModelName,
        modelVersion=version,
        label="test-sst",
    )

    ray = RayTracing(telescopeModel=tel, configData=configData)
    ray.simulate(test=True, force=True)
    ray.analyze(force=True)


@pytest.mark.skipif(not has_db_connection(), reason=DB_CONNECTION_MSG)
@pytest.mark.skipif(not simtel_installed(), reason=SIMTEL_MSG)
def test_rx():
    version = "current"
    label = "test-lst"

    configData = {
        "sourceDistance": 10 * u.km,
        "zenithAngle": 20 * u.deg,
        "offAxisAngle": [0, 2.5, 5.0] * u.deg,
    }

    tel = TelescopeModel(
        site="north", telescopeModelName="lst-1", modelVersion=version, label=label
    )

    ray = RayTracing(telescopeModel=tel, configData=configData)

    ray.simulate(test=True, force=True)
    ray_rx = copy(ray)

    ray.analyze(force=True)
    ray_rx.analyze(force=True, useRX=True)

    # Plotting d80
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel("off-axis")
    ax.set_ylabel("d80")

    ray.plot("d80_deg", marker="o", linestyle=":")
    ray_rx.plot("d80_deg", marker="s", linestyle="--")

    plotFilePSF = io.getTestPlotFile("d80_test_rx.pdf")
    plt.savefig(plotFilePSF)

    # Plotting effArea
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel("off-axis")
    ax.set_ylabel("eff. area")

    ray.plot("eff_area", marker="o", linestyle=":")
    ray_rx.plot("d80_deg", marker="s", linestyle="--")

    plotFileArea = io.getTestPlotFile("effArea_test_rx.pdf")
    plt.savefig(plotFileArea)


@pytest.mark.skipif(not has_db_connection(), reason=DB_CONNECTION_MSG)
@pytest.mark.skipif(not simtel_installed(), reason=SIMTEL_MSG)
def test_plot_image():
    version = "prod3"
    label = "test-astri"
    configData = {
        "sourceDistance": 10 * u.km,
        "zenithAngle": 20 * u.deg,
        "offAxisAngle": [0, 2.5, 5.0] * u.deg,
    }

    tel = TelescopeModel(
        site="south", telescopeModelName="sst-D", modelVersion=version, label=label
    )

    ray = RayTracing(telescopeModel=tel, configData=configData)

    ray.simulate(test=True, force=True)
    ray.analyze(force=True)

    # Plotting images
    for ii, image in enumerate(ray.images()):
        plt.figure(figsize=(8, 6), tight_layout=True)
        ax = plt.gca()
        ax.set_xlabel("X [cm]")
        ax.set_ylabel("Y [cm]")
        image.plotImage(psf_color="b")
        plotFile = io.getTestPlotFile("test_plot_image_{}.pdf".format(ii))
        plt.savefig(plotFile)


@pytest.mark.skipif(not has_db_connection(), reason=DB_CONNECTION_MSG)
@pytest.mark.skipif(not simtel_installed(), reason=SIMTEL_MSG)
def test_single_mirror(plot=False):

    # Test MST, single mirror PSF simulation
    version = "prod3"
    configData = {"mirrorNumbers": list(range(1, 5)), "singleMirrorMode": True}

    tel = TelescopeModel(
        site="north",
        telescopeModelName="mst-FlashCam-D",
        modelVersion=version,
        label="test-mst",
    )

    ray = RayTracing(telescopeModel=tel, configData=configData)
    ray.simulate(test=True, force=True)
    ray.analyze(force=True)

    # Plotting d80 histogram
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel("d80")

    ray.plotHistogram("d80_cm", color="r", bins=10)
    plotFile = io.getTestPlotFile("d80_hist_test.pdf")
    plt.savefig(plotFile)


@pytest.mark.skipif(not has_db_connection(), reason=DB_CONNECTION_MSG)
@pytest.mark.skipif(not simtel_installed(), reason=SIMTEL_MSG)
def test_integral_curve():
    version = "prod4"
    label = "lst_integral"

    configData = {
        "sourceDistance": 10 * u.km,
        "zenithAngle": 20 * u.deg,
        "offAxisAngle": [0] * u.deg,
    }

    tel = TelescopeModel(
        site="north",
        telescopeModelName="mst-FlashCam-D",
        modelVersion=version,
        label=label,
    )

    ray = RayTracing(telescopeModel=tel, configData=configData)

    ray.simulate(test=True, force=True)
    ray.analyze(force=True)

    # Plotting cumulative curve for each image
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel("radius [cm]")
    ax.set_ylabel("relative intensity")
    for im in ray.images():
        im.plotCumulative(color="b")
    plotFile = io.getTestPlotFile("test_cumulative_psf.pdf")
    plt.savefig(plotFile)

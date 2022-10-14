#!/usr/bin/python3

import logging
from copy import copy

import astropy.units as u
import matplotlib.pyplot as plt
import pytest

import simtools.io_handler as io
from simtools.model.telescope_model import TelescopeModel
from simtools.ray_tracing import RayTracing

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.mark.parametrize("telescopeModelName", ["sst-1M", "sst-ASTRI", "sst-GCT"])
def test_ssts(set_simtools, telescopeModelName):
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


def test_rx(set_simtools):
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

    plotFilePSF = io.getOutputFile(fileName="d80_test_rx.pdf", dirType="plots", test=True)
    plt.savefig(plotFilePSF)

    # Plotting effArea
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel("off-axis")
    ax.set_ylabel("eff. area")

    ray.plot("eff_area", marker="o", linestyle=":")
    ray_rx.plot("d80_deg", marker="s", linestyle="--")

    plotFileArea = io.getOutputFile(fileName="effArea_test_rx.pdf", dirType="plots", test=True)
    plt.savefig(plotFileArea)


def test_plot_image(set_simtools):
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
        plotFile = io.getOutputFile(
            fileName="test_plot_image_{}.pdf".format(ii), dirType="plots", test=True
        )
        plt.savefig(plotFile)


def test_single_mirror(set_simtools, plot=False):

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
    plotFile = io.getOutputFile(fileName="d80_hist_test.pdf", dirType="plots", test=True)
    plt.savefig(plotFile)


def test_integral_curve(set_simtools):
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
    plotFile = io.getOutputFile(fileName="test_cumulative_psf.pdf", dirType="plots", test=True)
    plt.savefig(plotFile)

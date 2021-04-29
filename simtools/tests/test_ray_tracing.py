#!/usr/bin/python3

import logging
import matplotlib.pyplot as plt
from copy import copy

import astropy.units as u

import simtools.io_handler as io
from simtools.ray_tracing import RayTracing
from simtools.model.telescope_model import TelescopeModel

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_ssts(show=False):
    # Test with 3 SSTs
    version = 'prod3'
    configData = {
        'sourceDistance': 10 * u.km,
        'zenithAngle': 20 * u.deg,
        'offAxisAngle': [0, 1.0, 2.0, 3.0, 4.0] * u.deg
    }
    telTypes = ['sst-1M', 'sst-ASTRI', 'sst-GCT']
    telModels = list()
    rayTracing = list()
    for t in telTypes:
        tel = TelescopeModel(
            site='south',
            telescopeModelName=t,
            modelVersion=version,
            label='test-sst'
        )
        telModels.append(t)

        ray = RayTracing(
            telescopeModel=tel,
            sourceDistance=sourceDistance,
            zenithAngle=zenithAngle,
            offAxisAngle=offAxisAngle
        )
        ray.simulate(test=True, force=True)
        ray.analyze(force=True)

        rayTracing.append(ray)

    # Plotting
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel('off-axis')
    ax.set_ylabel('d80')

    for ray in rayTracing:
        ray.plot('d80_deg', marker='o', linestyle=':')

    plotFile = io.getTestPlotFile('d80_test_ssts.pdf')
    plt.savefig(plotFile)


def test_rx():
    sourceDistance = 10 * u.km
    version = 'current'
    label = 'test-lst'
    zenithAngle = 20 * u.deg
    offAxisAngle = [0, 2.5, 5.0] * u.deg

    tel = TelescopeModel(
        site='north',
        telescopeModelName='lst-1',
        modelVersion=version,
        label=label
    )

    ray = RayTracing(
        telescopeModel=tel,
        sourceDistance=sourceDistance,
        zenithAngle=zenithAngle,
        offAxisAngle=offAxisAngle
    )

    ray.simulate(test=True, force=True)
    ray_rx = copy(ray)

    ray.analyze(force=True)
    ray_rx.analyze(force=True, useRX=True)

    # Plotting d80
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel('off-axis')
    ax.set_ylabel('d80')

    ray.plot('d80_deg', marker='o', linestyle=':')
    ray_rx.plot('d80_deg', marker='s', linestyle='--')

    plotFilePSF = io.getTestPlotFile('d80_test_rx.pdf')
    plt.savefig(plotFilePSF)

    # Plotting effArea
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel('off-axis')
    ax.set_ylabel('eff. area')

    ray.plot('eff_area', marker='o', linestyle=':')
    ray_rx.plot('d80_deg', marker='s', linestyle='--')

    plotFileArea = io.getTestPlotFile('effArea_test_rx.pdf')
    plt.savefig(plotFileArea)
    return


def test_plot_image():
    sourceDistance = 10 * u.km
    version = 'prod3'
    label = 'test-astri'
    zenithAngle = 20 * u.deg
    offAxisAngle = [0, 2.5, 5.0] * u.deg

    tel = TelescopeModel(
        site='south',
        telescopeModelName='sst-D',
        modelVersion=version,
        label=label
    )

    ray = RayTracing(
        telescopeModel=tel,
        sourceDistance=sourceDistance,
        zenithAngle=zenithAngle,
        offAxisAngle=offAxisAngle
    )

    ray.simulate(test=True, force=True)
    ray.analyze(force=True)

    # Plotting images
    for ii, image in enumerate(ray.images()):
        plt.figure(figsize=(8, 6), tight_layout=True)
        ax = plt.gca()
        ax.set_xlabel('X [cm]')
        ax.set_ylabel('Y [cm]')
        image.plotImage(psf_color='b')
        plotFile = io.getTestPlotFile('test_plot_image_{}.pdf'.format(ii))
        plt.savefig(plotFile)
    return


def test_single_mirror(plot=False):

    # Test MST, single mirror PSF simulation
    version = 'prod3'

    tel = TelescopeModel(
        site='north',
        telescopeModelName='mst-FlashCam-D',
        modelVersion=version,
        label='test-mst'
    )

    ray = RayTracing(
        telescopeModel=tel,
        singleMirrorMode=True,
        mirrorNumbers=list(range(1, 5))
    )
    ray.simulate(test=True, force=True)
    ray.analyze(force=True)

    # Plotting d80 histogram
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel('d80')

    ray.plotHistogram('d80_cm', color='r', bins=10)
    plotFile = io.getTestPlotFile('d80_hist_test.pdf')
    plt.savefig(plotFile)


def test_integral_curve():
    sourceDistance = 10 * u.km
    version = 'prod4'
    label = 'lst_integral'
    zenithAngle = 20 * u.deg
    offAxisAngle = [0] * u.deg

    tel = TelescopeModel(
        site='north',
        telescopeModelName='mst-FlashCam-D',
        modelVersion=version,
        label=label
    )

    ray = RayTracing(
        telescopeModel=tel,
        sourceDistance=sourceDistance,
        zenithAngle=zenithAngle,
        offAxisAngle=offAxisAngle
    )

    ray.simulate(test=True, force=True)
    ray.analyze(force=True)

    # Plotting cumulative curve for each image
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel('radius [cm]')
    ax.set_ylabel('relative intensity')
    for im in ray.images():
        im.plotCumulative(color='b')
    plotFile = io.getTestPlotFile('test_cumulative_psf.pdf')
    plt.savefig(plotFile)


def test_config_data():

    label = 'test-config-data'
    version = 'prod4'

    configData = {
        'sourceDistance': 10 * u.km,
        'zenithAngle': 30 * u.deg,
        'offAxisAngle': [0, 2] * u.deg
    }

    tel = TelescopeModel(
        site='north',
        telescopeModelName='mst-FlashCam-D',
        modelVersion=version,
        label=label
    )

    ray = RayTracing(
        telescopeModel=tel,
        configData=configData
    )

    assert ray._configData['zenithAngle'] == 30
    assert len(ray._configData['offAxisAngle']) == 2


if __name__ == '__main__':

    # test_ssts()
    # test_rx()
    # test_single_mirror()
    # test_plot_image()
    # test_integral_curve()
    test_config_data()

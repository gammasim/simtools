#!/usr/bin/python3

import logging
import matplotlib.pyplot as plt
import numpy as np
from copy import copy

from simtools.util import config as cfg
from simtools.ray_tracing import RayTracing
from simtools.telescope_model import TelescopeModel

logging.getLogger().setLevel(logging.DEBUG)


def test_ssts(show=False):
    # Test with 3 SSTs
    sourceDistance = 10  # km
    site = 'south'
    version = 'prod4'
    zenithAngle = 20
    offAxisAngle = [0, 1.0, 2.0, 3.0, 4.0]

    telTypes = ['sst-1m', 'sst-astri', 'sst-gct']
    telModels = list()
    rayTracing = list()
    for t in telTypes:
        tel = TelescopeModel(
            telescopeType=t,
            site=site,
            version=version,
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

    if show:
        plt.show()


def test_rx(show=False):
    sourceDistance = 10
    site = 'south'
    version = 'prod4'
    label = 'test-astri'
    zenithAngle = 20
    offAxisAngle = [0, 2.5, 5.0]

    tel = TelescopeModel(
        telescopeType='astri',
        site=site,
        version=version,
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

    # Plotting PSF images
    for im in ray.images():
        print(im)
        plt.figure(figsize=(8, 6), tight_layout=True)
        ax = plt.gca()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # psf_* for PSF circle
        # image_* for histogram
        im.plotImage(psf_color='b')

        ax.set_aspect('equal', adjustable='datalim')
        if show:
            plt.show()

    # Plotting d80
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel('off-axis')
    ax.set_ylabel('d80')

    ray.plot('d80_deg', marker='o', linestyle=':')
    ray_rx.plot('d80_deg', marker='s', linestyle='--')

    if show:
        plt.show()

    # Plotting effArea
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel('off-axis')
    ax.set_ylabel('eff. area')

    ray.plot('eff_area', marker='o', linestyle=':')
    ray_rx.plot('d80_deg', marker='s', linestyle='--')

    if show:
        plt.show()


def test_single_mirror(plot=False):

    # Test MST, single mirror PSF simulation
    site = 'south'
    version = 'prod4'

    tel = TelescopeModel(
        telescopeType='mst-flashcam',
        site=site,
        version=version,
        label='test-mst'
    )

    ray = RayTracing(
        telescopeModel=tel,
        singleMirrorMode=True,
        mirrorNumbers=list(range(1, 5))
    )
    ray.simulate(test=True, force=True)
    ray.analyze(force=True)

    # Plotting
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel('d80')

    ray.plotHistogram('d80_cm', color='r', bins=10)
    # ray.plot('d80_deg', color='r', linestyle='none', marker='o')

    if plot:
        plt.show()


def test_integral_curve(plot=False):
    sourceDistance = 10
    site = 'south'
    version = 'prod4'
    label = 'lst_integral'
    zenithAngle = 20
    offAxisAngle = [0]
    show = True

    tel = TelescopeModel(
        telescopeType='lst',
        site=site,
        version=version,
        label=label
    )

    ray = RayTracing(
        telescopeModel=tel,
        sourceDistance=sourceDistance,
        zenithAngle=zenithAngle,
        offAxisAngle=offAxisAngle
    )

    ray.simulate(test=True, force=False)
    ray.analyze(force=True)

    # Plotting PSF images
    for im in ray.images():
        print(im)
        plt.figure(figsize=(8, 6), tight_layout=True)
        ax = plt.gca()
        ax.set_xlabel('radius [cm]')
        ax.set_ylabel('')

        # psf_* for PSF circle
        # image_* for histogram
        im.plotIntegral(color='b')

        if show:
            plt.show()

    # Plotting d80
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel('off-axis')
    ax.set_ylabel('d80')

    ray.plot('d80_cm', marker='o', linestyle=':')

    if show:
        plt.show()


if __name__ == '__main__':

    # test_ssts(False)
    # test_rx(False)
    # test_single_mirror(False)
    # test_integral_curve(True)
    pass

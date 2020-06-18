#!/usr/bin/python3

import logging
import matplotlib.pyplot as plt

import simtools.io_handler as io
from simtools.model.telescope_model import TelescopeModel
from simtools.camera_efficiency import CameraEfficiency

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_main():
    tel = TelescopeModel(
        telescopeType='lst',
        site='south',
        version='p3',
        label='test_camera_eff'
    )
    ce = CameraEfficiency(telescopeModel=tel)
    ce.simulate(force=True)
    ce.analyze(force=True)

    # Plotting Cherenkov
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel('wl')
    ax.set_ylabel('eff')

    ce.plot('cherenkov')

    plotFileCherenkov = io.getTestPlotFile('camera_eff_cherenkov.pdf')
    plt.savefig(plotFileCherenkov)

    # Plotting NSB
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xlabel('wl')
    ax.set_ylabel('eff')

    ce.plot('nsb')

    plotFileNSB = io.getTestPlotFile('camera_eff_nsb.pdf')
    plt.savefig(plotFileNSB)


if __name__ == '__main__':

    test_main()
    pass

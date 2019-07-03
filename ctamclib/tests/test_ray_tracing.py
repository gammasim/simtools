#!/usr/bin/python3

import logging
import matplotlib.pyplot as plt
import numpy as np

from ctamclib.ray_tracing import RayTracing
from ctamclib.telescope_model import TelescopeModel

logging.getLogger().setLevel(logging.DEBUG)

if __name__ == '__main__':

    yamlDBPath = (
        '/home/prado/Work/Projects/CTA_MC/svn/Simulations/MCModelDescription/trunk/configReports'
    )
    simtelPath = (
        '/afs/ifh.de/group/cta/scratch/prado/corsika_simtelarray/corsika6.9_simtelarray_19-03-08'
    )

    sourceDistance = 10
    site = 'south'
    version = 'prod4'
    label = 'test-sst'
    zenithAngle = 20
    # offAxisAngle = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    offAxisAngle = [0, 1.0, 2.0, 3.0, 4.0, 5.0]

    telTypes = ['sst-1m', 'sst-astri', 'sst-gct']
    telModels = list()
    rayTracing = list()
    for t in telTypes:
        tel = TelescopeModel(
            yamlDBPath=yamlDBPath,
            telescopeType=t,
            site=site,
            version=version,
            label=label
        )
        telModels.append(t)

        ray = RayTracing(
            simtelSourcePath=simtelPath,
            telescopeModel=tel,
            sourceDistance=sourceDistance,
            zenithAngle=zenithAngle,
            offAxisAngle=offAxisAngle
        )
        ray.simulate(test=False, force=True)
        ray.analyze(force=True)

        rayTracing.append(ray)

    # Plotting

    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    ax.set_xlabel('off-axis')
    ax.set_ylabel('d80')

    for ray in rayTracing:
        ray.plot('d80_deg', marker='o', linestyle=':')

    plt.show()

    # script = simtel.getRunBashScript()

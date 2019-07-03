#!/usr/bin/python3

import logging

from ctamclib.ray_tracing import RayTracing
from ctamclib.telescope_model import TelescopeModel

logging.getLogger().setLevel(logging.DEBUG)

if __name__ == '__main__':

    yamlDBPath = (
        '/home/prado/Work/Projects/CTA_MC/svn/Simulations/MCModelDescription/trunk/configReports'
    )

    tel = TelescopeModel(
        yamlDBPath=yamlDBPath,
        telescopeType='lst',
        site='south',
        version='prod4',
        label='test-simtel'
    )

    simtelPath = (
        '/afs/ifh.de/group/cta/scratch/prado/corsika_simtelarray/corsika6.9_simtelarray_19-03-08'
    )

    rayTracing = RayTracing(
        simtelSourcePath=simtelPath,
        telescopeModel=tel,
        sourceDistance=10,
        zenithAngle=20,
        offAxisAngle=[0, 1, 2, 3]
    )

    print('TEST::', rayTracing)

    rayTracing.configParameters(zenithAngle=35, offAxisAngle=[0, 1, 2, 3], sourceDistance=10)

    rayTracing.simulate(test=True)

    # script = simtel.getRunBashScript()

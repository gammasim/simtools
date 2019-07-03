#!/usr/bin/python3

import logging

from ctamclib.simtel_runner import SimtelRunner
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

    simtel = SimtelRunner(
        simtelSourcePath=simtelPath,
        mode='ray-tracing',
        telescopeModel=tel,
        zenithAngle=20,
        offAxisAngle=2
    )

    print('TEST:', simtel)

    # simtel.configParameters(zenithAngle=20, offAxisAngle=2, sourceDistance=10)

    simtel.run(test=True)

    # script = simtel.getRunBashScript()

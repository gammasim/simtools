#!/usr/bin/python3

import logging

from ctamclib.simtel_runner import SimtelRunner
from ctamclib.telescope_model import TelescopeModel

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)

yamlDBPath = (
    '/home/prado/Work/Projects/CTA_MC/svn/Simulations/MCModelDescription/trunk/configReports'
)
simtelPath = (
    '/afs/ifh.de/group/cta/scratch/prado/corsika_simtelarray/corsika6.9_simtelarray_19-03-08'
)


def test_ray_tracing_mode():
    tel = TelescopeModel(
        yamlDBPath=yamlDBPath,
        telescopeType='lst',
        site='south',
        version='prod4',
        label='test-simtel'
    )

    simtel = SimtelRunner(
        simtelSourcePath=simtelPath,
        mode='ray-tracing',
        telescopeModel=tel,
        zenithAngle=20,
        offAxisAngle=2
    )

    logger.info(simtel)

    simtel.run(test=True, force=True)


if __name__ == '__main__':

    test_ray_tracing_mode()

#!/usr/bin/python3

import logging

import simtools.io_handler as io
from simtools.simtel.simtel_events import SimtelEvents

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_reading_files():
    files = list()
    files.append(io.getTestDataFile(
        'run201_proton_za20deg_azm0deg-North-Prod5_test-production-5-mini.simtel.zst')
    )
    files.append(io.getTestDataFile(
        'run202_proton_za20deg_azm0deg-North-Prod5_test-production-5-mini.simtel.zst')
    )

    print(files)

    simtel_events = SimtelEvents(inputFiles=files)


if __name__ == '__main__':
    test_reading_files()

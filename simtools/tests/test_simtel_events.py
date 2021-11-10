#!/usr/bin/python3

import logging

import simtools.io_handler as io
from simtools.simtel.simtel_events import SimtelEvents

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_reading_files():
    files = list()
    files.append(io.getTestDataFile(
        'run1_gamma_za20deg_azm0deg-North-Prod5_test-production-5.simtel.zst')
    )
    files.append(io.getTestDataFile(
        'run2_gamma_za20deg_azm0deg-North-Prod5_test-production-5.simtel.zst')
    )

    print(files)

    simtel_events = SimtelEvents(files=files)


if __name__ == '__main__':
    test_reading_files()

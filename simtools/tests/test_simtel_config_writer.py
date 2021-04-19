#!/usr/bin/python3

import logging
import unittest

from simtools.simtel.simtel_config_writer import SimtelConfigWriter

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TestSimtelConfigWriter(unittest.TestCase):

    def setUp():
        self.simtelConfigWriter = SimtelConfigWriter()

    def test_write_array_config_file():
        pass


if __name__ == '__main__':
    # unittest.main()

    tt = TestSimtelConfigWriter()
    tt.setUp()
    tt.test_write_array_config_file()

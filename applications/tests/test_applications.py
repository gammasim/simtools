#!/usr/bin/python3

import logging
import unittest
import os

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TestApplications(unittest.TestCase):

    def setUp(self):
        pass

    def test_applications(self):
        os.system('python applications/produce_array_config.py')


if __name__ == '__main__':
    unittest.main()

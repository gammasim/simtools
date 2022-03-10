#!/usr/bin/python3

import logging
import unittest

from simtools.simtel.simtel_runner import SimtelRunner, SimtelExecutionError

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TestSimtelRunner(unittest.TestCase):
    def setUp(self):
        self.simtelRunner = SimtelRunner()

    def test_run(self):
        with self.assertRaises(RuntimeError):
            self.simtelRunner.run()

    def test_simtel_execution_error(self):
        with self.assertRaises(SimtelExecutionError):
            self.simtelRunner._raiseSimtelError()


if __name__ == "__main__":
    unittest.main()

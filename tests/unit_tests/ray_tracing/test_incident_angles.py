#!/usr/bin/python3

"""
Unit tests for the incident_angles module.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

import astropy.units as u
from astropy.table import QTable

from simtools.ray_tracing.incident_angles import IncidentAnglesCalculator


class TestIncidentAnglesCalculator(unittest.TestCase):
    """Test the IncidentAnglesCalculator class."""

    def setUp(self):
        """Set up the test."""
        self.config_file = "test_config.cfg"
        self.tel_id = 1
        self.output_dir = Path("test_output")
        self.ray_tracing_config = "test_ray_tracing.cfg"
        self.number_of_rays = 100

        # Create a mock instance
        self.calculator = IncidentAnglesCalculator(
            config_file=self.config_file,
            tel_id=self.tel_id,
            output_dir=self.output_dir,
            ray_tracing_config=self.ray_tracing_config,
            number_of_rays=self.number_of_rays,
        )

    @patch("simtools.ray_tracing.ray_tracing.RayTracing")
    def test_initialization(self, mock_ray_tracing):
        """Test initialization of the calculator."""
        # Check that attributes are set correctly
        assert self.calculator.config_file == self.config_file
        assert self.calculator.tel_id == self.tel_id
        assert self.calculator.output_dir == self.output_dir
        assert self.calculator.ray_tracing_config == self.ray_tracing_config
        assert self.calculator.number_of_rays == self.number_of_rays
        assert self.calculator.results is None

    @patch("simtools.ray_tracing.ray_tracing.RayTracing")
    def test_run(self, mock_ray_tracing):
        """Test the run method."""
        # Mock the ray tracing results
        mock_instance = mock_ray_tracing.return_value
        mock_instance.run.return_value = "ray_tracing_output.txt"

        # Create mock ray tracing data
        mock_data = {
            "x_pix": [0, 1, 2],
            "y_pix": [0, 1, 2],
            "incident_angle_deg": [10, 20, 30],
        }

        # Mock the parse_ray_tracing_output method
        with patch.object(self.calculator, "_parse_ray_tracing_output", return_value=mock_data):
            self.calculator.run()

            # Check that ray tracing was called with the correct parameters
            mock_ray_tracing.assert_called_once_with(
                telescope_model_file=self.config_file,
                telescope_model_name=self.tel_id,
                number_of_rays=self.number_of_rays,
                ray_tracing_config_file=self.ray_tracing_config,
                output_path=self.output_dir,
            )

            # Check that the result was saved
            assert isinstance(self.calculator.results, QTable)
            assert len(self.calculator.results) == 3
            assert "x_pix" in self.calculator.results.colnames
            assert "y_pix" in self.calculator.results.colnames
            assert "incident_angle" in self.calculator.results.colnames

    @patch("simtools.ray_tracing.ray_tracing.RayTracing")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_incident_angles(self, mock_savefig, mock_ray_tracing):
        """Test the plot_incident_angles method."""
        # Mock data in self.results
        self.calculator.results = QTable()
        self.calculator.results["x_pix"] = [0, 1, 2]
        self.calculator.results["y_pix"] = [0, 1, 2]
        self.calculator.results["incident_angle"] = [10, 20, 30] * u.deg

        # Call the plot method
        self.calculator.plot_incident_angles()

        # Check that savefig was called
        mock_savefig.assert_called()


if __name__ == "__main__":
    unittest.main()

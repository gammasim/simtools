"""Runner for the packaged obdeect optical ray tracer."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import astropy.units as u

from simtools import settings


class SimulatorObdeect:
    """Run ``obdeect-simtools-raytrace`` for one ray-tracing configuration.

    It runs a provenance-bound optical model exported by
    ``obdeect.optical_model_compiler``.
    """

    def __init__(
        self,
        telescope_model,
        label=None,
        config_data=None,
        output_file=None,
        force_simulate=False,
        test=False,
    ):
        self._logger = logging.getLogger(__name__)
        self.telescope_model = telescope_model
        self.label = label or getattr(telescope_model, "label", "obdeect")
        self.config = dict(config_data or {})
        self.output_file = Path(output_file) if output_file is not None else None
        self.force_simulate = force_simulate
        self.photons_per_run = int(self.config.get("number_of_photons", 100 if test else 10000))
        if self.photons_per_run < 1:
            raise ValueError("number_of_photons must be positive")
        if self.output_file is None:
            raise ValueError("output_file is required for obdeect simulations")
        if not self.config.get("obdeect_optical_model_file"):
            raise ValueError("obdeect_optical_model_file is required for model-derived ray tracing")
        if not Path(self.config["obdeect_optical_model_file"]).expanduser().is_file():
            raise FileNotFoundError("obdeect_optical_model_file does not exist")
        if self.config.get("single_mirror_mode", False):
            raise ValueError("single_mirror_mode is not supported by the reference obdeect CLI")

    @staticmethod
    def _value(value, unit):
        """Convert an astropy quantity or scalar to a float in ``unit``."""
        if isinstance(value, u.Quantity):
            return float(value.to_value(unit))
        return float(value)

    def make_run_command(self):
        """Build the packaged native command line."""
        distance_m = self._value(self.config.get("source_distance", 10.0), u.km) * 1000.0
        command = [str(settings.config.obdeect_exe)]
        optical_model_file = self.config.get("obdeect_optical_model_file")
        command.extend(["--optical-model", str(Path(optical_model_file).expanduser())])
        command.extend(
            [
                "--source",
                str(self.config.get("source", "star")),
                "--photons",
                str(self.photons_per_run),
                "--output",
                str(self.output_file),
                "--field-x-deg",
                str(float(self.config.get("off_axis_x", 0.0))),
                "--field-y-deg",
                str(float(self.config.get("off_axis_y", 0.0))),
                "--distance-m",
                str(distance_m),
                "--wavelength-nm",
                str(float(self.config.get("wavelength_nm", 400.0))),
            ]
        )
        for key in ("source_x_m", "source_y_m", "source_z_m", "divergence_deg"):
            if key in self.config:
                command.extend([f"--{key.replace('_', '-')}", str(float(self.config[key]))])
        for key in ("direction_x", "direction_y", "direction_z"):
            if key in self.config:
                command.extend([f"--{key.replace('_', '-')}", str(float(self.config[key]))])
        return command

    def run(self, test=False):  # pylint: disable=unused-argument
        """Execute the native tracer and validate that it produced arrivals."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        if self.output_file.exists() and not self.force_simulate:
            return
        command = self.make_run_command()
        self._logger.info("Running obdeect ray tracer: %s", command)
        subprocess.run(command, check=True)
        if not self.output_file.is_file() or self.output_file.stat().st_size == 0:
            raise RuntimeError(f"obdeect did not write an arrival file: {self.output_file}")

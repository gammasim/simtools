"""Runner for the packaged obdeect optical ray tracer."""

from __future__ import annotations

import logging
import math
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
        self._imaging_metadata = None
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
        metadata = self._load_imaging_metadata(distance_m)
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
                str(self.output_file.with_suffix(".obdeect.csv")),
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
        field_x, field_y = self._source_field_angles()
        command[command.index("--field-x-deg") + 1] = str(field_x)
        command[command.index("--field-y-deg") + 1] = str(field_y)
        command.extend(
            [
                "--focal-surface-image",
                "--star-mode",
                "finite",
                "--source-seed",
                "0",
                "--launch-radius-m",
                str(metadata.launch_radius_m),
                "--entrance-z-m",
                str(metadata.entrance_z_m),
            ]
        )
        for key in ("source_x_m", "source_y_m", "source_z_m", "divergence_deg"):
            if key in self.config:
                command.extend([f"--{key.replace('_', '-')}", str(float(self.config[key]))])
        for key in ("direction_x", "direction_y", "direction_z"):
            if key in self.config:
                command.extend([f"--{key.replace('_', '-')}", str(float(self.config[key]))])
        return command

    def _load_imaging_metadata(self, distance_m):
        """Resolve source and camera metadata at the backend boundary."""
        from obdeect.imaging_list import load_imaging_metadata  # pylint: disable=import-outside-toplevel

        if self.config.get("source", "star") != "star":
            raise ValueError("simtools imaging studies require a star source")
        self._imaging_metadata = load_imaging_metadata(
            Path(self.config["obdeect_optical_model_file"]).expanduser(),
            distance_m,
            float(self.config.get("off_axis_x", 0)),
            float(self.config.get("off_axis_y", 0)),
            self._value(self.config.get("zenith_angle", 0), u.deg),
        )
        return self._imaging_metadata

    def _source_field_angles(self):
        """Transform the sim_telarray star and telescope pointing into local slopes."""
        zenith = math.radians(self._value(self.config.get("zenith_angle", 0), u.deg))
        theta = zenith - math.radians(float(self.config.get("off_axis_x", 0)))
        phi = 0.0
        if zenith > math.radians(1e-6):
            phi = math.radians(float(self.config.get("off_axis_y", 0))) / math.sin(zenith)
        # Source unit vector in the tilted telescope basis; ray propagation is its negative.
        sx = math.sin(zenith) * math.cos(theta) * math.cos(phi) - math.cos(zenith) * math.sin(theta)
        sy = math.sin(zenith) * math.sin(phi)
        sz = math.sin(zenith) * math.sin(theta) * math.cos(phi) + math.cos(zenith) * math.cos(theta)
        return math.degrees(math.atan2(-sx, sz)), math.degrees(math.atan2(-sy, sz))

    def _write_imaging_list(self, native_output):
        """Expose native optical arrivals using the shared analysis input contract."""
        from obdeect.imaging_list import write_imaging_list  # pylint: disable=import-outside-toplevel

        write_imaging_list(
            native_output, self.output_file, self._imaging_metadata, self.photons_per_run
        )
        native_output.unlink()

    def run(self, test=False):  # pylint: disable=unused-argument
        """Execute the native tracer and validate that it produced arrivals."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        if self.output_file.exists() and not self.force_simulate:
            return
        command = self.make_run_command()
        self._logger.info("Running obdeect ray tracer: %s", command)
        subprocess.run(command, check=True)
        native_output = self.output_file.with_suffix(".obdeect.csv")
        if not native_output.is_file() or native_output.stat().st_size == 0:
            raise RuntimeError(f"obdeect did not write an arrival file: {native_output}")
        self._write_imaging_list(native_output)

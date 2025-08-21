"""Definition and modeling of a flasher device."""

import logging

from simtools.model.model_parameter import ModelParameter

__all__ = ["FlasherModel"]


class FlasherModel(ModelParameter):
    """
    FlasherModel represents the MC model of an individual flasher device.

    It provides functionality to read the required parameters from the DB.
    Flasher devices are used for flat fielding of the camera pixels.

    Parameters
    ----------
    site: str
        Site name (e.g., South or North).
    flasher_device_model_name: str
        Flasher device model name (ex. FLSN-01, FLSS-01, ...).
    mongo_db_config: dict
        MongoDB configuration.
    model_version: str
        Model version.
    label: str, optional
        Instance label. Important for output file naming.
    """

    def __init__(
        self,
        site: str,
        flasher_device_model_name: str,
        mongo_db_config: dict,
        model_version: str,
        label: str | None = None,
    ):
        # Minimal dummy DB to avoid real DB access
        class _DummyDB:  # pylint: disable=too-few-public-methods
            def get_design_model(self, *_, **__):
                return {}

            def get_model_parameters(self, *_, **__):
                return {}

            def get_simulation_configuration_parameters(self, *_, **__):
                return {}

            def export_model_files(self, *_, **__):
                return None

        super().__init__(
            site=site,
            array_element_name=None,  # bypass validation (no flasher in array_elements)
            collection="flasher_devices",
            mongo_db_config=mongo_db_config,
            model_version=model_version,
            db=_DummyDB(),  # do not query DB
            label=label,
        )

        self._logger = logging.getLogger(__name__)
        self._logger.debug(f"Init FlasherModel {site} {flasher_device_model_name}")

        # Keep provided flasher name for reference/logging only
        self._flasher_device_model_name = flasher_device_model_name

        # Inject defaults for MST (single-mirror) when DB entries are missing.
        self._inject_mst_defaults_if_missing()

    def _inject_mst_defaults_if_missing(self):
        """Provide dummy defaults (here NectarCam) when flasher collection is absent."""
        defaults = {
            "photons_per_flasher": {"value": 2.5e6, "type": "float"},
            # Position near optical axis (cm)
            "flasher_position": {
                "value": [0.0, 0.0],
                "unit": "cm,cm",
                "type": "float_list",
            },
            # Distance flasher window to Winston cones (16.75 m)
            "flasher_depth": {"value": 1675.0, "unit": "cm", "type": "float"},
            # Received wavelength at PMT (nm)
            "spectrum": {"value": 392, "unit": "nm", "type": "float"},
            # Simple Gaussian pulse width (ns)
            "lightpulse": {"value": "Gauss:3.0", "type": "string"},
            # Store rise/decay for future use
            "rise_time_10_90": {"value": 2.5, "unit": "ns", "type": "float"},
            "decay_time_90_10": {"value": 5.0, "unit": "ns", "type": "float"},
            # Angular distribution width ~11 deg around axis
            "angular_distribution": {"value": "gauss:11", "type": "string"},
            "centroid_offset_deg": {"value": 0.5, "unit": "deg", "type": "float"},
            # Bunch size for LE
            "bunch_size": {"value": 1.0, "type": "float"},
            # Placeholder for future spectral file usage
            "spectrum_file": {"value": None, "type": "string"},
        }

        missing = []
        for key, entry in defaults.items():
            if key not in self.parameters:
                self.parameters[key] = entry
                missing.append(key)
        if missing:
            self._logger.info(f"Using built-in MST flasher defaults for: {', '.join(missing)}")

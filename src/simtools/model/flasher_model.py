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
        """Initialize FlasherModel."""
        super().__init__(
            site=site,
            array_element_name=flasher_device_model_name,
            collection="flasher_devices",
            mongo_db_config=mongo_db_config,
            model_version=model_version,
            db=None,
            label=label,
        )

        self._logger = logging.getLogger(__name__)
        self._logger.debug(f"Init FlasherModel {site} {flasher_device_model_name}")

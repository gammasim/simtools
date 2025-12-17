"""Definition and modeling of a calibration device."""

import logging

from simtools.model.model_parameter import ModelParameter


class CalibrationModel(ModelParameter):
    """
    CalibrationModel represents the MC model of an individual calibration device.

    It provides functionality to read the required parameters from the DB.

    Parameters
    ----------
    site: str
        Site name (e.g., South or North).
    calibration_device_model_name: str
        Calibration device model name (ex. ILLS-01, ILLN-01, ...).
    model_version: str
        Model version.
    label: str, optional
        Instance label. Important for output file naming.
    overwrite_model_parameters: str, optional
        File name to overwrite model parameters from DB with provided values.
    """

    def __init__(
        self,
        site,
        calibration_device_model_name,
        model_version,
        label=None,
        overwrite_model_parameters=None,
    ):
        """Initialize CalibrationModel."""
        super().__init__(
            site=site,
            array_element_name=calibration_device_model_name,
            collection="calibration_devices",
            model_version=model_version,
            label=label,
            overwrite_model_parameters=overwrite_model_parameters,
        )

        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CalibrationModel %s %s", site, calibration_device_model_name)

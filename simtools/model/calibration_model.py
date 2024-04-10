import logging

from simtools.model.model_parameter import ModelParameter

__all__ = ["CalibrationModel"]


class CalibrationModel(ModelParameter):
    """
    CalibrationModel represents the MC model of an individual calibration device. \
    It contains the list of parameters that can be read from the DB.

    Parameters
    ----------
    site: str
        Site name (e.g., South or North).
    calibratrion_model_name: str
        Calibration deivce model name (ex. ILLS-01, ILLN-01, ...).
    mongo_db_config: dict
        MongoDB configuration.
    model_version: str
        Version of the model (ex. prod5).
    label: str
        Instance label. Important for output file naming.
    """

    def __init__(
        self,
        site,
        calibration_device_model_name,
        mongo_db_config,
        model_version,
        label=None,
    ):
        """
        Initialize TelescopeModel.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CalibrationModel %s %s", site, calibration_device_model_name)
        ModelParameter.__init__(
            self,
            site=site,
            telescope_model_name=calibration_device_model_name,
            mongo_db_config=mongo_db_config,
            model_version=model_version,
            db=None,
            label=label,
        )

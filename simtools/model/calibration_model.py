import logging
from pydoc import locate

from astropy import units as u

from simtools import db_handler
from simtools.io_operations import io_handler
from simtools.utils import names

__all__ = ["InvalidParameter", "CalibrationModel"]


class InvalidParameter(Exception):
    """Exception for invalid parameter."""


class CalibrationModel:
    """
    CalibrationModel represents the model of a calibration device. It contains the list of \
    parameters that can be read from the DB. A set of methods are available to ...

    Parameters
    ----------
    site: str
        South or North.
    calibration_device_name: str
        Calibration device name (ex. ILLN-01, ...).
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
        calibration_device_name,
        mongo_db_config=None,
        model_version="Released",
        db=None,
        label=None,
    ):
        """
        Initialize CalibrationModel.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init CalibrationModel")

        self.site = names.validate_site_name(site)
        # TODO add validate function
        self.name = names.validate_telescope_model_name(calibration_device_name)
        self.model_version = names.validate_model_version_name(model_version)
        self.label = label
        self._extra_label = None

        self.io_handler = io_handler.IOHandler()
        self.db = None
        if db is not None:
            self.db = db
        elif mongo_db_config is not None:
            self._logger.debug("Connecting to DB")
            self.db = db_handler.DatabaseHandler(mongo_db_config=mongo_db_config)

        self._parameters = {}

        self._load_parameters_from_db()

        self._set_config_file_directory_and_name()
        self._is_config_file_up_to_date = False
        self._is_exported_model_files_up_to_date = False

    @property
    def extra_label(self):
        """
        Return the extra label if defined, if not return ''.
        """
        return self._extra_label if self._extra_label is not None else ""

    def _set_config_file_directory_and_name(self):
        """Define the variable _config_file_directory and create directories, if needed."""

        self._config_file_directory = self.io_handler.get_output_directory(
            label=self.label, sub_dir="model"
        )

        # Setting file name and the location
        config_file_name = names.simtel_telescope_config_file_name(
            self.site, self.name, self.model_version, self.label, self._extra_label
        )
        self._config_file_path = self._config_file_directory.joinpath(config_file_name)

    def _load_parameters_from_db(self):
        """Read parameters from DB and store them in _parameters."""

        if self.db is None:
            return

        self._logger.debug("Reading calibration device parameters from DB")

        self._set_config_file_directory_and_name()
        self._parameters = self.db.get_model_parameters(
            self.site, self.name, self.model_version, only_applicable=True
        )

        self._logger.debug("Reading site parameters from DB")
        _site_pars = self.db.get_site_parameters(
            self.site, self.model_version, only_applicable=True
        )
        self._parameters.update(_site_pars)

    def has_parameter(self, par_name):
        """
        Verify if the parameter is in the model.

        Parameters
        ----------
        par_name: str
            Name of the parameter.

        Returns
        -------
        bool
            True if parameter is in the model.
        """
        return par_name in self._parameters

    def get_parameter(self, par_name):
        """
        Get an existing parameter of the model, including derived parameters.

        Parameters
        ----------
        par_name: str
            Name of the parameter.

        Returns
        -------
        Value of the parameter

        Raises
        ------
        InvalidParameter
            If par_name does not match any parameter in this model.
        """
        try:
            return self._parameters[par_name]
        except KeyError as e:
            msg = f"Parameter {par_name} was not found in the model"
            self._logger.error(msg)
            raise InvalidParameter(msg) from e

    def get_parameter_value(self, par_name):
        """
        Get the value of an existing parameter of the model.

        Parameters
        ----------
        par_name: str
            Name of the parameter.

        Returns
        -------
        Value of the parameter.

        Raises
        ------
        InvalidParameter
            If par_name does not match any parameter in this model.
        """
        par_info = self.get_parameter(par_name)
        return par_info["Value"]

    def get_parameter_value_with_unit(self, par_name):
        """
        Get the value of an existing parameter of the model as an Astropy Quantity with its unit.\
        If no unit is provided in the model, the value is returned without a unit.

        Parameters
        ----------
        par_name: str
            Name of the parameter.

        Returns
        -------
        Astropy quantity with the value of the parameter multiplied by its unit. If no unit is \
        provided in the model, the value is returned without a unit.

        Raises
        ------
        InvalidParameter
            If par_name does not match any parameter in this model.
        """
        par_info = self.get_parameter(par_name)
        if "units" in par_info:
            return par_info["Value"] * u.Unit(par_info["units"])
        return par_info["Value"]

    def add_parameter(self, par_name, value, is_file=False, is_aplicable=True):
        """
        Add a new parameters to the model. This function does not modify the DB, it affects only \
        the current instance.

        Parameters
        ----------
        par_name: str
            Name of the parameter.
        value:
            Value of the parameter.
        is_file: bool
            Indicates whether the new parameter is a file or not.
        is_aplicable: bool
            Indicates whether the new parameter is applicable or not.

        Raises
        ------
        InvalidParameter
            If an existing parameter is tried to be added.
        """
        if par_name in self._parameters:
            msg = f"Parameter {par_name} already in the model, use change_parameter instead"
            self._logger.error(msg)
            raise InvalidParameter(msg)

        self._logger.info(f"Adding {par_name}={value} to the model")
        self._parameters[par_name] = {}
        self._parameters[par_name]["Value"] = value
        self._parameters[par_name]["Type"] = type(value)
        self._parameters[par_name]["Applicable"] = is_aplicable
        self._parameters[par_name]["File"] = is_file

        self._is_config_file_up_to_date = False
        if is_file:
            self._is_exported_model_files_up_to_date = False

    def change_parameter(self, par_name, value):
        """
        Change the value of an existing parameter to the model. This function does not modify the \
        DB, it affects only the current instance.

        Parameters
        ----------
        par_name: str
            Name of the parameter.
        value:
            Value of the parameter.

        Raises
        ------
        InvalidParameter
            If the parameter to be changed does not exist in this model.
        """
        if par_name not in self._parameters:
            msg = f"Parameter {par_name} not in the model, use add_parameters instead"
            self._logger.error(msg)
            raise InvalidParameter(msg)

        type_of_par_name = locate(self._parameters[par_name]["Type"])
        if not isinstance(value, type_of_par_name):
            self._logger.warning(
                f"The type of the provided value ({value}, {type(value)}) "
                f"is different from the type of {par_name} "
                f"({self._parameters[par_name]['Type']}). "
                f"Attempting to cast to the correct type."
            )
            try:
                value = type_of_par_name(value)
            except ValueError:
                self._logger.error(
                    f"Could not cast {value} to {self._parameters[par_name]['Type']}."
                )
                raise

        self._logger.debug(
            f"Changing parameter {par_name} "
            f"from {self._parameters[par_name]['Value']} to {value}"
        )
        self._parameters[par_name]["Value"] = value

        # In case parameter is a file, the model files will be outdated
        if self._parameters[par_name]["File"]:
            self._is_exported_model_files_up_to_date = False

        self._is_config_file_up_to_date = False

    def change_multiple_parameters(self, **kwargs):
        """
        Change the value of multiple existing parameters in the model. This function does not \
        modify the DB, it affects only the current instance.

        Parameters
        ----------
        **kwargs
            Parameters should be passed as parameter_name=value.

        Raises
        ------
        InvalidParameter
            If at least one of the parameters to be changed does not exist in this model.
        """
        for par, value in kwargs.items():
            if par in self._parameters:
                self.change_parameter(par, value)
            else:
                self.add_parameter(par, value)

        self._is_config_file_up_to_date = False

    def remove_parameters(self, *args):
        """
        Remove a set of parameters from the model.

        Parameters
        ----------
        *args
            Each parameter to be removed has to be passed as args.

        Raises
        ------
        InvalidParameter
            If at least one of the parameter to be removed is not in this model.
        """
        for par in args:
            if par in self._parameters:
                self._logger.info(f"Removing parameter {par}")
                del self._parameters[par]
            else:
                msg = f"Could not remove parameter {par} because it does not exist"
                self._logger.error(msg)
                raise InvalidParameter(msg)
        self._is_config_file_up_to_date = False

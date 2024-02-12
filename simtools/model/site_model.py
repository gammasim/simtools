#!/usr/bin/python3

import logging

from simtools.model.model_parameter import ModelParameter

__all__ = ["SiteModel"]


class SiteModel(ModelParameter):
    """
    SiteModel represents the MC model of an observatory site.

    Parameters
    ----------
    site: str
        Site name (e.g., South or North).
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
        mongo_db_config=None,
        model_version="Released",
        db=None,
        label=None,
    ):
        """
        Initialize SiteModel
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SiteModel")
        ModelParameter.__init__(
            self,
            site=site,
            mongo_db_config=mongo_db_config,
            model_version=model_version,
            db=db,
            label=label,
        )

    def get_reference_point(self):
        """
        Get reference point coordinates as dict

        Returns
        -------
        dict
            Reference point coordinates as dict
        """

        return {
            "center_altitude": self.get_parameter_value_with_unit("reference_point_altitude"),
            "center_northing": self.get_parameter_value_with_unit("reference_point_utm_north"),
            "center_easting": self.get_parameter_value_with_unit("reference_point_utm_east"),
            "epsg_code": self.get_parameter_value("epsg_code"),
        }

    def get_corsika_site_parameters(self, config_file_style=False):
        """
        Get site-related CORSIKA parameters as dict.
        Parameters are returned with units wherever possible.

        Parameters
        ----------
        config_file_style bool
            Return using style of corsika_parameters.yml file

        Returns
        -------
        dict
            Site-related CORSIKA parameters as dict

        """

        # backwards compatibility to `corsika_parameters.yml` (temporary)
        if config_file_style:
            _atmosphere_id = 26 if self.site == "North" else 36
            return {
                "OBSLEV": [self.get_parameter_value("corsika_observation_level")],
                "ATMOSPHERE": [_atmosphere_id, "Y"],
                "MAGNET": [
                    self.get_parameter_value("geomag_horizontal"),
                    self.get_parameter_value("geomag_vertical"),
                ],
                "ARRANG": [self.get_parameter_value("geomag_rotation")],
            }

        return {
            "corsika_observation_level": self.get_parameter_value_with_unit(
                "corsika_observation_level"
            ),
            "geomag_horizontal": self.get_parameter_value_with_unit("geomag_horizontal"),
            "geomag_vertical": self.get_parameter_value_with_unit("geomag_vertical"),
            "geomag_rotation": self.get_parameter_value_with_unit("geomag_rotation"),
        }

    def get_simtel_parameters(self, telescope_model=False, site_model=True):
        """
        Get simtel site parameters as dict

        Returns
        -------
        dict
            Simtel site parameters as dict

        """
        return super().get_simtel_parameters(telescope_model=telescope_model, site_model=site_model)

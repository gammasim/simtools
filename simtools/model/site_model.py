#!/usr/bin/python3
"""Definition of site model."""

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
        mongo_db_config,
        model_version,
        label=None,
    ):
        """Initialize SiteModel."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SiteModel for site %s", site)
        ModelParameter.__init__(
            self,
            site=site,
            mongo_db_config=mongo_db_config,
            model_version=model_version,
            db=None,
            label=label,
        )

    def get_reference_point(self):
        """
        Get reference point coordinates as dict.

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
        # backwards compatibility to `corsika_parameters.yml` (temporary TODO)
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

    def get_array_elements_for_layout(self, layout_name):
        """
        Return list of array elements for a given array layout.

        Parameters
        ----------
        layout_name: str
            Name of the array layout

        Returns
        -------
        list
            List of array elements
        """
        layouts = self.get_parameter_value("array_layouts")
        for layout in layouts:
            if layout["name"] == layout_name.lower():
                return layout["elements"]
        self._logger.error(
            "Array layout '%s' not found in '%s' site model.", layout_name, self.site
        )
        raise ValueError

    def get_list_of_array_layouts(self):
        """
        Get list of available array layouts.

        Returns
        -------
        list
            List of available array layouts
        """
        return [layout["name"] for layout in self.get_parameter_value("array_layouts")]

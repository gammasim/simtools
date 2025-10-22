#!/usr/bin/python3
"""Definition of site model."""

import logging
from pathlib import Path

import numpy as np
from astropy import units as u

from simtools.model.model_parameter import ModelParameter


class SiteModel(ModelParameter):
    """
    Representation of an observatory site model.

    The site model includes (among others):

    - Reference point coordinates
    - Array elements
    - Geomagnetic field parameters
    - Atmospheric parameters
    - NSB parameters

    Parameters
    ----------
    site: str
        Site name (e.g., South or North).
    db_config: dict
        Database configuration.
    model_version: str or list
        Model version or list of model versions (in which case only the first one is used).
    label: str, optional
        Instance label.
    overwrite_model_parameters: str, optional
        File name to overwrite model parameters from DB with provided values.
    ignore_software_version: bool, optional
        If True, ignore software version checks for deprecated parameters.
    """

    def __init__(
        self,
        site,
        db_config,
        model_version,
        label=None,
        overwrite_model_parameters=None,
        ignore_software_version=False,
    ):
        """Initialize SiteModel."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SiteModel for site %s", site)
        super().__init__(
            site=site,
            db_config=db_config,
            model_version=model_version,
            label=label,
            collection="sites",
            overwrite_model_parameters=overwrite_model_parameters,
            ignore_software_version=ignore_software_version,
        )

    def get_reference_point(self):
        """
        Get reference point coordinates.

        Ground coordinates are calculated relative to this point.

        Returns
        -------
        dict
            Reference point coordinates.
        """
        return {
            "center_altitude": self.get_parameter_value_with_unit("reference_point_altitude"),
            "center_northing": self.get_parameter_value_with_unit("reference_point_utm_north"),
            "center_easting": self.get_parameter_value_with_unit("reference_point_utm_east"),
            "epsg_code": self.get_parameter_value("epsg_code"),
        }

    def get_corsika_site_parameters(self, config_file_style=False, model_directory=None):
        """
        Get site-related CORSIKA parameters.

        Parameters are returned with units wherever possible ('config_file_style=False')
        or in CORSIKA-expected coordinates.

        Parameters
        ----------
        config_file_style: bool
            Return using CORSIKA config file syntax
        model_directory: Path, optional
            Model directory to use for file paths

        Returns
        -------
        dict
            Site-related CORSIKA parameters.
        """
        if config_file_style:
            model_directory = model_directory or Path()
            return {
                "OBSLEV": [
                    self.get_parameter_value_with_unit("corsika_observation_level").to_value("cm")
                ],
                # We always use a custom profile by filename, so this has to be set to 99
                "ATMOSPHERE": [99, "Y"],
                "IACT ATMOFILE": [
                    model_directory / self.get_parameter_value("atmospheric_profile")
                ],
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
        raise ValueError(f"Array layout '{layout_name}' not found in '{self.site}' site model.")

    def get_array_elements_of_type(self, array_element_type):
        """
        Get all array elements of a given type.

        Parameters
        ----------
        array_element_type : str
            Type of the array element (e.g. LSTN, MSTS)

        Returns
        -------
        dict
            Dict with array elements.
        """
        return self.db.get_array_elements_of_type(
            array_element_type=array_element_type,
            model_version=self.model_version,
            collection="telescopes",
        )

    def get_list_of_array_layouts(self):
        """
        Get list of available array layouts.

        Returns
        -------
        list
            List of available array layouts
        """
        return [layout["name"] for layout in self.get_parameter_value("array_layouts")]

    def export_atmospheric_transmission_file(self, model_directory):
        """
        Export atmospheric transmission file from database to the given directory.

        Parameters
        ----------
        model_directory: Path
            Model directory to export the file to.
        """
        self.db.export_model_files(
            parameters={
                "atmospheric_transmission_file": {
                    "value": self.get_parameter_value("atmospheric_profile"),
                    "file": True,
                }
            },
            dest=model_directory,
        )

    def get_nsb_integrated_flux(self, wavelength_min=300 * u.nm, wavelength_max=650 * u.nm):
        """
        Get the integrated flux for the NSB (Night Sky Background) model.

        Returns
        -------
        float
            Integrated flux value.
        """
        table = self.db.get_ecsv_file_as_astropy_table(
            file_name=self.get_parameter_value("nsb_spectrum")
        )
        table.sort("wavelength")
        wl = table["wavelength"].quantity.to(u.nm)
        rate = table["differential photon rate"].quantity.to(1 / (u.nm * u.cm**2 * u.ns * u.sr))
        mask = (wl >= wavelength_min) & (wl <= wavelength_max)
        integral_cm2 = np.trapezoid(rate[mask], wl[mask])
        self._logger.debug(
            f"NSB integral between {wavelength_min} and {wavelength_max}: {integral_cm2}"
        )
        return integral_cm2.value

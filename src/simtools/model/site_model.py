#!/usr/bin/python3
"""Definition of site model."""

import logging
from pathlib import Path

import numpy as np
from astropy import units as u

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
    model_version: str or list
        Model version or list of model versions (in which case only the first one is used).
    label: str, optional
        Instance label. Important for output file naming.
    """

    def __init__(
        self,
        site: str,
        mongo_db_config: dict,
        model_version: str,
        label: str | None = None,
    ):
        """Initialize SiteModel."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Init SiteModel for site %s", site)
        super().__init__(
            site=site,
            mongo_db_config=mongo_db_config,
            model_version=model_version,
            db=None,
            label=label,
            collection="sites",
        )

    def get_reference_point(self) -> dict:
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

    def get_corsika_site_parameters(
        self, config_file_style: bool = False, model_directory: Path | None = None
    ) -> dict:
        """
        Get site-related CORSIKA parameters as dict.

        Parameters are returned with units wherever possible.

        Parameters
        ----------
        config_file_style: bool
            Return using CORSIKA config file syntax
        model_directory: Path, optional
            Model directory to use for file paths

        Returns
        -------
        dict
            Site-related CORSIKA parameters as dict
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

    def get_array_elements_for_layout(self, layout_name: str) -> list:
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

    def get_list_of_array_layouts(self) -> list:
        """
        Get list of available array layouts.

        Returns
        -------
        list
            List of available array layouts
        """
        return [layout["name"] for layout in self.get_parameter_value("array_layouts")]

    def export_atmospheric_transmission_file(self, model_directory: Path):
        """
        Export atmospheric transmission file.

        This method is needed because when CORSIKA is not piped to sim_telarray,
        the atmospheric transmission file is not written out to the model directory.
        This method allows to export it explicitly.

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

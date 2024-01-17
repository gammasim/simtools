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
            "center_lat": self.get_parameter_value_with_unit("reference_point_latitude"),
            "center_lon": self.get_parameter_value_with_unit("reference_point_longitude"),
            "center_alt": self.get_parameter_value_with_unit("reference_point_altitude"),
            "EPSG": self.get_parameter_value("epsg_code"),
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

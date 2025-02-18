"""Estimates compute and storage requirements for a simulation production."""

import astropy.units as u
import numpy as np
import yaml

from simtools.utils import names


class ResourceEstimator:
    """
    Estimates compute and storage resources required for simulations.

    The estimation can be based on historical data from existing simulations or
    by making guesses per event using the number of events provided.

    Attributes
    ----------
    grid_point : dict
        Dictionary containing parameters such as energy (TeV), azimuth (deg),
        zenith (deg), and night sky background.
    simulation_params : dict
        Dictionary containing simulation parameters, including the number of events and site.
    existing_data : list of dict, optional
        List of dictionaries with historical data of compute and storage
          resources for existing simulations.

    Methods
    -------
    estimate_resources() -> dict:
        Estimates resources required for the simulation using the number of
        events from simulation_params.
    interpolate_resources() -> dict:
        Interpolates resources based on existing data.
    guess_resources_per_event() -> dict:
        Estimates resources per event using grid point parameters.
    """

    def __init__(
        self,
        grid_point: dict[str, float],
        simulation_params: dict[str, float],
        existing_data: list[dict] | None = None,
        lookup_file: str = "production_resource_estimates.yaml",
    ):
        """
        Initialize the resource estimator.

        Parameters
        ----------
        grid_point : dict
            Dictionary containing parameters such as energy (TeV), azimuth (deg),
            zenith (deg), and night sky background.
        simulation_params : dict
            Dictionary containing simulation parameters, including the number of events and site.
        existing_data : list of dict, optional
            List of dictionaries with historical data of compute and storage
            resources for existing simulations.
        lookup_file : str, optional
            Path to the YAML file containing the lookup table.
        """
        self.grid_point = grid_point
        self.simulation_params = simulation_params
        self.existing_data = existing_data or []
        self.lookup_table = self.load_lookup_table(lookup_file)

        self.site = names.validate_site_name(self.simulation_params["site"])

    @staticmethod
    def load_lookup_table(lookup_file: str) -> dict:
        """
        Load the lookup table from a YAML file.

        Parameters
        ----------
        lookup_file : str
            Path to the YAML file containing the lookup table.

        Returns
        -------
        dict
            Dictionary containing the lookup table.
        """
        with open(lookup_file, encoding="utf-8") as file:
            lookup = yaml.safe_load(file)

        # return as quantities
        for _, data in lookup.items():
            for _, resources in data["Zenith"].items():
                resources["compute_per_event"] = u.Quantity(
                    resources["compute_per_event"]["value"], resources["compute_per_event"]["unit"]
                )
                resources["storage_per_event"] = u.Quantity(
                    resources["storage_per_event"]["value"], resources["storage_per_event"]["unit"]
                )

        return lookup

    def estimate_resources(self) -> dict:
        """
        Estimate the compute and storage resources required for the simulation.

        Returns
        -------
        dict
            A dictionary with estimates for compute and storage resources, with units.
        """
        number_of_events = self.simulation_params.get("number_of_events")
        if self.existing_data:
            return self.interpolate_resources(number_of_events)

        return self.guess_resources_per_event(number_of_events)

    def interpolate_resources(self, number_of_events: int) -> dict:
        """
        Interpolate resources required for the simulation from existing data.

        Parameters
        ----------
        number_of_events : int
            The number of events for which to interpolate resources.

        Returns
        -------
        dict
            A dictionary with interpolated estimates for compute and storage resources, with units.
        """
        azimuth = self.grid_point["azimuth"]
        zenith = self.grid_point["zenith"]
        nsb = self.grid_point["night_sky_background"]

        closest_data = min(
            self.existing_data,
            key=lambda x: (
                abs(x["azimuth"] - azimuth) + abs(x["zenith"] - zenith) + abs(x["nsb"] - nsb)
            ),
        )

        compute_total_value = closest_data["compute_total"]["value"]
        compute_total_unit = closest_data["compute_total"]["unit"]

        storage_total_value = closest_data["storage_total"]["value"]
        storage_total_unit = closest_data["storage_total"]["unit"]

        compute_total = (
            compute_total_value
            * (number_of_events / closest_data["events"])
            * u.Unit(compute_total_unit)
        )
        storage_total = (
            storage_total_value
            * (number_of_events / closest_data["events"])
            * u.Unit(storage_total_unit)
        )

        return {"compute_total": compute_total, "storage_total": storage_total}

    def guess_resources_per_event(self, number_of_events: int) -> dict:
        """
        Estimate resources for the simulation based on grid point parameters and per-event guess.

        Parameters
        ----------
        number_of_events : int
            The number of events for which to estimate resources.

        Returns
        -------
        dict
            A dictionary with guessed estimates for compute and storage resources, with units.
        """
        zenith = self.grid_point["zenith"]
        zeniths = sorted(self.lookup_table[self.site]["Zenith"].keys())

        if zenith <= zeniths[0]:
            compute_per_event = self.lookup_table[self.site]["Zenith"][zeniths[0]][
                "compute_per_event"
            ]
            storage_per_event = self.lookup_table[self.site]["Zenith"][zeniths[0]][
                "storage_per_event"
            ]
        elif zenith >= zeniths[-1]:
            compute_per_event = self.lookup_table[self.site]["Zenith"][zeniths[-1]][
                "compute_per_event"
            ]
            storage_per_event = self.lookup_table[self.site]["Zenith"][zeniths[-1]][
                "storage_per_event"
            ]
        else:
            lower_bound = max(e for e in zeniths if e <= zenith)
            upper_bound = min(e for e in zeniths if e >= zenith)
            lower_values = self.lookup_table[self.site]["Zenith"][lower_bound]
            upper_values = self.lookup_table[self.site]["Zenith"][upper_bound]

            compute_per_event = (
                np.interp(
                    zenith,
                    [lower_bound, upper_bound],
                    [
                        lower_values["compute_per_event"].value,
                        upper_values["compute_per_event"].value,
                    ],
                )
                * lower_values["compute_per_event"].unit
            )

            storage_per_event = (
                np.interp(
                    zenith,
                    [lower_bound, upper_bound],
                    [
                        lower_values["storage_per_event"].value,
                        upper_values["storage_per_event"].value,
                    ],
                )
                * lower_values["storage_per_event"].unit
            )

        compute = number_of_events * compute_per_event
        storage = number_of_events * storage_per_event

        return {"compute": compute.to("h"), "storage": storage.to("GB")}

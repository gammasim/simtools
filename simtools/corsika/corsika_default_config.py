from collections import defaultdict

import astropy.units as u
from scipy.interpolate import interp1d


class CorsikaDefaultConfig:
    """
    This class contains the default configuration for CORSIKA parameters for
    the various primary particles. It includes all basic dependencies on zenith angles, etc.
    The default values defined in this class assume the full CTAO arrays are simulated,
    including full CTAO energy range and number of events optimized to run for roughly 24 hours
    on a single node on the grid.
    """

    def __init__(self, primary=None, zenith_angle=None):
        """
        Initialize CorsikaDefaultConfig.
        """
        self.primary = primary
        self.zenith_angle = zenith_angle
        self._energy_slope = -2.0
        self.energy_ranges = self._define_hardcoded_energy_ranges()
        self.number_of_showers = self._define_hardcoded_number_of_showers()

    @property
    def primary(self):
        """
        Primary particle.

        Returns
        -------
        primary: str
        """
        return self._primary

    @primary.setter
    def primary(self, primary):
        """
        Set primary particle.

        Parameters
        ----------
        primary: str
            Which primary to simulate.
        """
        supported_primaries = [
            "gamma",
            "gamma_diffuse",
            "electron",
            "proton",
            "muon",
            "helium",
            "nitrogen",
            "silicon",
            "iron",
        ]
        if primary not in supported_primaries:
            raise ValueError(
                f"Invalid primary particle: {primary}. Must be one of {supported_primaries}"
            )
        self._primary = primary

    @property
    def zenith_angle(self):
        """
        Zenith angle.

        Returns
        -------
        zenith_angle: astropy.units.Quantity
        """
        return self._zenith_angle

    @zenith_angle.setter
    @u.quantity_input(zenith_angle=u.deg)
    def zenith_angle(self, zenith_angle):
        """
        Set zenith angle.

        Parameters
        ----------
        zenith_angle: astropy.units.Quantity
            Which zenith angle to simulate (in degrees).
        """
        allowed_zenith_angle_interval = [20.0, 60.0] * u.deg
        if (
            zenith_angle < allowed_zenith_angle_interval[0]
            or zenith_angle > allowed_zenith_angle_interval[1]
        ):
            raise ValueError(
                f"The zenith angle, {zenith_angle:.1f}, "
                f"is outside of the allowed interval, {allowed_zenith_angle_interval}."
                "This interval is enforced because values can only be interpolated "
                "between zenith angles that were manually optimized."
            )

        self._zenith_angle = zenith_angle

    @property
    def energy_slope(self):
        """
        Energy slope.

        Returns
        -------
        energy_slope: float
        """
        return self._energy_slope

    def _define_hardcoded_energy_ranges(self):
        """
        Define the hardcoded energy ranges for the various primaries.
        These energy ranges are for the full CTAO energy range (for both sites).

        Returns
        -------
        energy_ranges: dict
            Dictionary with the default energy ranges for the various primaries.
        """

        energy_ranges = defaultdict(dict)
        energy_ranges["gamma"][20] = [3 * u.GeV, 330 * u.TeV]
        energy_ranges["gamma"][40] = [6 * u.GeV, 660 * u.TeV]
        energy_ranges["gamma"][60] = [12 * u.GeV, 990 * u.TeV]

        for zenith_angle in energy_ranges["gamma"]:
            energy_ranges["gamma_diffuse"][zenith_angle] = energy_ranges["gamma"][zenith_angle]
            energy_ranges["electron"][zenith_angle] = energy_ranges["gamma"][zenith_angle]

        energy_ranges["proton"][20] = [8 * u.GeV, 600 * u.TeV]
        energy_ranges["proton"][40] = [12 * u.GeV, 800 * u.TeV]
        energy_ranges["proton"][60] = [16 * u.GeV, 1800 * u.TeV]

        energy_ranges["helium"][20] = [10 * u.GeV, 1200 * u.TeV]
        energy_ranges["helium"][40] = [20 * u.GeV, 2400 * u.TeV]
        energy_ranges["helium"][60] = [40 * u.GeV, 3600 * u.TeV]

        energy_ranges["nitrogen"][20] = [40 * u.GeV, 4000 * u.TeV]
        energy_ranges["nitrogen"][40] = [80 * u.GeV, 8000 * u.TeV]
        energy_ranges["nitrogen"][60] = [160 * u.GeV, 12000 * u.TeV]

        energy_ranges["silicon"][20] = [50 * u.GeV, 5000 * u.TeV]
        energy_ranges["silicon"][40] = [100 * u.GeV, 10000 * u.TeV]
        energy_ranges["silicon"][60] = [200 * u.GeV, 15000 * u.TeV]

        energy_ranges["iron"][20] = [60 * u.GeV, 6000 * u.TeV]
        energy_ranges["iron"][40] = [120 * u.GeV, 12000 * u.TeV]
        energy_ranges["iron"][60] = [240 * u.GeV, 18000 * u.TeV]

        return energy_ranges

    def _define_hardcoded_number_of_showers(self):
        """
        Define the hardcoded number of showers for the various primaries.

        Returns
        -------
        number_of_showers: dict
            Dictionary with the default number of showers for the various primaries.
        """

        number_of_showers = defaultdict(dict)
        number_of_showers["gamma"][20] = 5000
        number_of_showers["gamma"][40] = 5000
        number_of_showers["gamma"][60] = 2500

        for zenith_angle in number_of_showers["gamma"]:
            number_of_showers["gamma_diffuse"][zenith_angle] = number_of_showers["gamma"][
                zenith_angle
            ]
            number_of_showers["electron"][zenith_angle] = number_of_showers["gamma"][zenith_angle]

        number_of_showers["proton"][20] = 15000
        number_of_showers["proton"][40] = 15000
        number_of_showers["proton"][60] = 7500

        number_of_showers["helium"][20] = 10000
        number_of_showers["helium"][40] = 10000
        number_of_showers["helium"][60] = 5000

        number_of_showers["nitrogen"][20] = 2000
        number_of_showers["nitrogen"][40] = 2000
        number_of_showers["nitrogen"][60] = 1000

        number_of_showers["silicon"][20] = 1000
        number_of_showers["silicon"][40] = 1000
        number_of_showers["silicon"][60] = 500

        return number_of_showers

    @staticmethod
    @u.quantity_input(zenith_angle=u.deg)
    def interpolate_to_zenith_angle(
        zenith_angle, zenith_angles_to_interpolate, values_to_interpolate
    ):
        """
        Interpolate values like energy range or number of showers to the provided zenith angle.

        Parameters
        ----------
        zenith_angle: astropy.units.Quantity
            Which zenith angle to interpolate to (in degrees).
        zenith_angles_to_interpolate: list
            List of zenith angles for which we have values to interpolate between.
        values_to_interpolate: list
            List of values to interpolate between.

        Returns
        -------
        float
            Interpolated value.
        """
        interpolation_function = interp1d(
            zenith_angles_to_interpolate, values_to_interpolate, kind="quadratic"
        )
        return interpolation_function(zenith_angle.to_value(u.deg)).item()

    def energy_range_for_primary(self):
        """
        Get the energy range for the primary particle for the given zenith angle.

        Returns
        -------
        energy_range: list
            List with the energy range for the primary particle for the given zenith angle.
        """

        zenith_angles_to_interpolate = [*self.energy_ranges[self.primary].keys()]
        min_energy_to_interpolate = [
            energy[0].to_value(u.GeV) for energy in self.energy_ranges[self.primary].values()
        ]
        max_energy_to_interpolate = [
            energy[1].to_value(u.GeV) for energy in self.energy_ranges[self.primary].values()
        ]

        return [
            self.interpolate_to_zenith_angle(
                self.zenith_angle, zenith_angles_to_interpolate, min_energy_to_interpolate
            )
            * u.GeV,
            (
                self.interpolate_to_zenith_angle(
                    self.zenith_angle, zenith_angles_to_interpolate, max_energy_to_interpolate
                )
                * u.GeV
            ).to(u.TeV),
        ]

    def number_of_showers_for_primary(self):
        """
        Get the number of showers for the primary particle for the given zenith angle.

        Returns
        -------
        number_of_showers: int
            Number of showers for the primary particle for the given zenith angle.
        """

        zenith_angles_to_interpolate = [*self.energy_ranges[self.primary].keys()]
        number_of_showers = [*self.number_of_showers[self.primary].values()]

        return self.interpolate_to_zenith_angle(
            self.zenith_angle,
            zenith_angles_to_interpolate,
            number_of_showers,
        )

    def view_cone_for_primary(self):
        """
        Get the view cone for the primary particle.
        All diffuse primaries have a view cone of 10 deg by default.

        Returns
        -------
        view_cone: list
            List with the view cone for the primary particle.
        """
        if self.primary == "gamma":
            return [0.0 * u.deg, 0.0 * u.deg]

        return [0 * u.deg, 10 * u.deg]

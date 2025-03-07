"""Calculate the thresholds for energy, radial distance, and viewcone."""

import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import AltAz
from ctapipe.coordinates import GroundFrame, TiltedGroundFrame


class LimitCalculator:
    """
    Compute thresholds/limits for energy, radial distance, and viewcone.

    Event data is read from the generated HDF5 file.

    Parameters
    ----------
    event_data_file : str
        Path to the HDF5 file containing the event data.
    telescope_list : list, optional
        List of telescope IDs to filter the events (default is None).
    """

    def __init__(self, event_data_file, telescope_list=None):
        """
        Initialize the LimitCalculator with the given event data file.

        Parameters
        ----------
        event_data_file : str
            Path to the HDF5 file containing the event data.
        telescope_list : list, optional
            List of telescope IDs to filter the events (default is None).
        """
        self.event_data_file = event_data_file
        self.telescope_list = telescope_list
        self.event_x_core = None
        self.event_y_core = None
        self.simulated = None
        self.shower_id_triggered = None
        self.list_of_files = None
        self.shower_sim_azimuth = None
        self.shower_sim_altitude = None
        self.array_azimuth = None
        self.array_altitude = None
        self.trigger_telescope_list_list = None
        self.units = {}
        self._read_event_data()

    def _read_event_data(self):
        """Read the event data from the HDF5 file."""
        with h5py.File(self.event_data_file, "r") as f:
            if "data" in f:
                data_group = f["data"]
                try:
                    self.event_x_core = data_group["core_x"][:]
                    self.event_y_core = data_group["core_y"][:]
                    self.simulated = data_group["simulated"][:]
                    self.shower_id_triggered = data_group["shower_id_triggered"][:]
                    self.list_of_files = data_group["file_names"][:]
                    self.shower_sim_azimuth = data_group["shower_sim_azimuth"][:]
                    self.shower_sim_altitude = data_group["shower_sim_altitude"][:]
                    self.array_altitude = data_group["array_altitudes"][:]
                    self.array_azimuth = data_group["array_azimuths"][:]
                    self.trigger_telescope_list_list = data_group["trigger_telescope_list_list"][:]
                except KeyError as exc:
                    raise KeyError(
                        "One or more required datasets are missing from the 'data' group."
                    ) from exc
                try:
                    self.units["core_x"] = data_group["core_x"].attrs.get("units", None)
                    self.units["core_y"] = data_group["core_y"].attrs.get("units", None)
                    self.units["simulated"] = data_group["simulated"].attrs.get("units", None)
                    self.units["shower_id_triggered"] = data_group["shower_id_triggered"].attrs.get(
                        "units", None
                    )
                    self.units["shower_sim_azimuth"] = data_group["shower_sim_azimuth"].attrs.get(
                        "units", None
                    )
                    self.units["shower_sim_altitude"] = data_group["shower_sim_altitude"].attrs.get(
                        "units", None
                    )
                    self.units["array_altitudes"] = data_group["array_altitudes"].attrs.get(
                        "units", None
                    )
                    self.units["array_azimuths"] = data_group["array_azimuths"].attrs.get(
                        "units", None
                    )
                except KeyError:
                    # Units are optional
                    pass
            else:
                raise KeyError("data group is missing from the HDF5 file.")

    def _compute_limits(self, hist, bin_edges, loss_fraction, limit_type="lower"):
        """
        Compute the limits based on the loss fraction.

        Parameters
        ----------
        hist : np.ndarray
            1D histogram array.
        bin_edges : np.ndarray
            Array of bin edges.
        loss_fraction : float
            Fraction of events to be lost.
        limit_type : str, optional
            Type of limit ('lower' or 'upper'). Default is 'lower'.

        Returns
        -------
        float
            Bin edge value corresponding to the threshold.
        """
        cumulative_sum = np.cumsum(hist) if limit_type == "upper" else np.cumsum(hist[::-1])
        total_events = np.sum(hist)
        threshold = (1 - loss_fraction) * total_events
        bin_index = np.searchsorted(cumulative_sum, threshold)
        return bin_edges[bin_index] if limit_type == "upper" else bin_edges[-bin_index]

    def _prepare_data_for_limits(self):
        """
        Prepare the data required for computing limits.

        Returns
        -------
        tuple
            Tuple containing core distances, triggered energies, core bins, and energy bins.
        """
        num_files = len(self.list_of_files)
        showers_per_file = len(self.simulated) // num_files
        shower_id_triggered_adjusted = self.adjust_shower_ids_across_files(
            self.shower_id_triggered, num_files, showers_per_file
        )

        if self.telescope_list is not None:
            mask = np.array(
                [
                    all(tel in event for tel in self.telescope_list)
                    for event in self.trigger_telescope_list_list
                ]
            )
            shower_id_triggered_adjusted = shower_id_triggered_adjusted[mask]

        triggered_energies = self.simulated[shower_id_triggered_adjusted]
        energy_bins = np.logspace(
            np.log10(triggered_energies.min()), np.log10(triggered_energies.max()), 1000
        )
        event_x_core_shower, event_y_core_shower = self._transform_to_shower_coordinates()
        core_distances_all = np.sqrt(event_x_core_shower**2 + event_y_core_shower**2)
        core_distances_triggered = core_distances_all[shower_id_triggered_adjusted]
        core_bins = np.linspace(
            core_distances_triggered.min(), core_distances_triggered.max(), 1000
        )

        return core_distances_triggered, triggered_energies, core_bins, energy_bins

    def compute_lower_energy_limit(self, loss_fraction):
        """
        Compute the lower energy limit in TeV based on the event loss fraction.

        Parameters
        ----------
        loss_fraction : float
            Fraction of events to be lost.

        Returns
        -------
        astropy.units.Quantity
            Lower energy limit.
        """
        _, triggered_energies, _, energy_bins = self._prepare_data_for_limits()

        hist, _ = np.histogram(triggered_energies, bins=energy_bins)
        lower_bin_edge_value = self._compute_limits(
            hist, energy_bins, loss_fraction, limit_type="lower"
        )
        return lower_bin_edge_value * u.TeV

    def compute_upper_radial_distance(self, loss_fraction):
        """
        Compute the upper radial distance based on the event loss fraction.

        Parameters
        ----------
        loss_fraction : float
            Fraction of events to be lost.

        Returns
        -------
        astropy.units.Quantity
            Upper radial distance in m.
        """
        core_distances_triggered, _, core_bins, _ = self._prepare_data_for_limits()

        hist, _ = np.histogram(core_distances_triggered, bins=core_bins)
        upper_bin_edge_value = self._compute_limits(
            hist, core_bins, loss_fraction, limit_type="upper"
        )
        return upper_bin_edge_value * u.m

    def compute_viewcone(self, loss_fraction):
        """
        Compute the viewcone based on the event loss fraction.

        Parameters
        ----------
        loss_fraction : float
            Fraction of events to be lost.

        Returns
        -------
        astropy.units.Quantity
            Viewcone radius in degrees.
        """
        # already in radians
        azimuth_diff = self.array_azimuth - self.shower_sim_azimuth  # * (np.pi / 180.0)
        sim_altitude_rad = self.shower_sim_altitude  # * (np.pi / 180.0)
        array_altitude_rad = self.array_altitude  # * (np.pi / 180.0)

        x_1 = np.cos(azimuth_diff) * np.cos(sim_altitude_rad)
        y_1 = np.sin(azimuth_diff) * np.cos(sim_altitude_rad)
        z_1 = np.sin(sim_altitude_rad)
        x_2 = x_1 * np.sin(array_altitude_rad) - z_1 * np.cos(array_altitude_rad)
        y_2 = y_1
        z_2 = x_1 * np.cos(array_altitude_rad) + z_1 * np.sin(array_altitude_rad)
        off_angles = np.arctan2(np.sqrt(x_2**2 + y_2**2), z_2) * (180.0 / np.pi)

        # Convert to AltAz frame
        array_altaz = AltAz(az=self.array_azimuth * u.rad, alt=self.array_altitude * u.rad)
        shower_altaz = AltAz(
            az=self.shower_sim_azimuth * u.rad, alt=self.shower_sim_altitude * u.rad
        )

        # Calculate the separation angle
        off_angles2 = array_altaz.separation(shower_altaz).deg

        print(np.all(off_angles == off_angles2))

        angle_bins = np.linspace(off_angles.min(), off_angles.max(), 400)
        hist, _ = np.histogram(off_angles, bins=angle_bins)

        upper_bin_edge_value = self._compute_limits(
            hist, angle_bins, loss_fraction, limit_type="upper"
        )
        return upper_bin_edge_value * u.deg

    def _transform_to_shower_coordinates(self):
        """
        Transform core positions from ground coordinates to shower coordinates.

        Returns
        -------
        tuple
            Core positions in shower coordinates (x, y).
        """
        pointing_az = self.shower_sim_azimuth * u.rad
        pointing_alt = self.shower_sim_altitude * u.rad

        pointing = AltAz(az=pointing_az, alt=pointing_alt)
        ground = GroundFrame(x=self.event_x_core * u.m, y=self.event_y_core * u.m, z=0 * u.m)
        shower_frame = ground.transform_to(TiltedGroundFrame(pointing_direction=pointing))

        return shower_frame.x.value, shower_frame.y.value

    def plot_data(self):
        """Plot the core distances and energies of triggered events."""
        num_files = len(self.list_of_files)
        showers_per_file = len(self.simulated) // num_files
        shower_id_triggered_adjusted = self.adjust_shower_ids_across_files(
            self.shower_id_triggered, num_files, showers_per_file
        )

        if self.telescope_list is not None:
            mask = np.array(
                [
                    all(tel in event for tel in self.telescope_list)
                    for event in self.trigger_telescope_list_list
                ]
            )
            shower_id_triggered_adjusted = shower_id_triggered_adjusted[mask]

        core_distances_all = np.sqrt(self.event_x_core**2 + self.event_y_core**2)
        core_distances_triggered = core_distances_all[shower_id_triggered_adjusted]
        triggered_energies = self.simulated[shower_id_triggered_adjusted]

        core_bins = np.linspace(core_distances_triggered.min(), core_distances_triggered.max(), 400)
        energy_bins = np.logspace(
            np.log10(triggered_energies.min()), np.log10(triggered_energies.max()), 400
        )
        plt.figure(figsize=(8, 6))
        plt.hist2d(
            core_distances_triggered,
            triggered_energies,
            bins=[core_bins, energy_bins],
            norm="log",
            cmap="viridis",
        )

        plt.colorbar(label="Event Count")
        plt.xlabel("Core Distance [m]")
        plt.ylabel("Energy [TeV]")
        plt.yscale("log")
        plt.title("2D Histogram of Triggered Core Distance vs Energy")
        plt.show()

    @staticmethod
    def adjust_shower_ids_across_files(shower_id_triggered, num_files, showers_per_file):
        """
        Adjust shower_id_triggered by incrementing the IDs for each file so they don't repeat.

        Parameters
        ----------
        shower_id_triggered : np.ndarray
            Array of shower IDs for triggered events.
        num_files : int
            Number of files.
        showers_per_file : int
            Number of showers per file.

        Returns
        -------
        np.ndarray
            Adjusted shower IDs.
        """
        increment_list = [i * showers_per_file for i in range(num_files)]
        return shower_id_triggered + np.repeat(increment_list, showers_per_file)

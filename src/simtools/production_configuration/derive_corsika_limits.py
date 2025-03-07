"""Calculate the thresholds for energy, radial distance, and viewcone."""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import tables
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
        with tables.open_file(self.event_data_file, mode="r") as f:
            reduced_data = f.root.data.reduced_data
            triggered_data = f.root.data.triggered_data
            file_names = f.root.data.file_names
            trigger_telescope_list_list = f.root.data.trigger_telescope_list_list

            self.event_x_core = reduced_data.col("core_x")
            self.event_y_core = reduced_data.col("core_y")
            self.simulated = reduced_data.col("simulated")
            self.shower_id_triggered = triggered_data.col("shower_id_triggered")
            self.list_of_files = file_names.col("file_names")
            self.shower_sim_azimuth = reduced_data.col("shower_sim_azimuth")
            self.shower_sim_altitude = reduced_data.col("shower_sim_altitude")
            self.array_altitude = reduced_data.col("array_altitudes")
            self.array_azimuth = reduced_data.col("array_azimuths")

            self.trigger_telescope_list_list = [
                [np.int16(tel) for tel in event] for event in trigger_telescope_list_list
            ]

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
        shower_id_triggered_masked = self.shower_id_triggered
        if self.telescope_list is not None:
            mask = np.array(
                [
                    all(tel in event for tel in self.telescope_list)
                    for event in self.trigger_telescope_list_list
                ]
            )
            shower_id_triggered_masked = self.shower_id_triggered[mask]

        triggered_energies = self.simulated[shower_id_triggered_masked]
        energy_bins = np.logspace(
            np.log10(triggered_energies.min()), np.log10(triggered_energies.max()), 1000
        )
        event_x_core_shower, event_y_core_shower = self._transform_to_shower_coordinates()
        core_distances_all = np.sqrt(event_x_core_shower**2 + event_y_core_shower**2)
        core_distances_triggered = core_distances_all[shower_id_triggered_masked]
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
        shower_id_triggered_masked = self.shower_id_triggered
        if self.telescope_list is not None:
            mask = np.array(
                [
                    all(tel in event for tel in self.telescope_list)
                    for event in self.trigger_telescope_list_list
                ]
            )
            shower_id_triggered_masked = self.shower_id_triggered[mask]

        core_distances_all = np.sqrt(self.event_x_core**2 + self.event_y_core**2)
        core_distances_triggered = core_distances_all[shower_id_triggered_masked]
        triggered_energies = self.simulated[shower_id_triggered_masked]

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

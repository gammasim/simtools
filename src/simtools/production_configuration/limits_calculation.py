"""Calculate the thresholds for energy, radial distance, and viewcone."""

import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np


class LimitCalculator:
    """
    Compute thresholds/limits for energy, radial distance, and viewcone.

    Event data is read from the generated HDF5 file.

    Parameters
    ----------
    event_data_file : str
        Path to the HDF5 file containing the event data.
    """

    def __init__(self, event_data_file):
        """
        Initialize the LimitCalculator with the given event data file.

        Parameters
        ----------
        event_data_file : str
            Path to the HDF5 file containing the event data.
        """
        self.event_data_file = event_data_file
        self.event_x_core = None
        self.event_y_core = None
        self.simulated = None
        self.shower_id_triggered = None
        self.list_of_files = None
        self.shower_azimuth = None
        self.shower_sim_altitude = None
        self.array_azimuth = None
        self.array_altitude = None
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
                    self.shower_azimuth = data_group["shower_azimuth"][:]
                    self.shower_sim_altitude = data_group["shower_sim_altitude"][:]
                    self.array_altitude = data_group["array_altitude"][:]
                except KeyError as exc:
                    raise KeyError(
                        "One or more required datasets are missing from the 'data' group."
                    ) from exc
            else:
                raise KeyError("'data' group is missing from the HDF5 file.")

    def _generate_2d_histogram(self, x_data, y_data, x_bins, y_bins):
        """
        Generate a 2D histogram from the given data.

        Parameters
        ----------
        x_data : np.ndarray
            Array of x-axis data.
        y_data : np.ndarray
            Array of y-axis data.
        x_bins : np.ndarray
            Array of bin edges for the x-axis.
        y_bins : np.ndarray
            Array of bin edges for the y-axis.

        Returns
        -------
        np.ndarray
            2D histogram array.
        """
        hist, _, _ = np.histogram2d(x_data, y_data, bins=[x_bins, y_bins])
        return hist

    def _compute_limits(self, hist, bin_edges, loss_fraction, axis=0, limit_type="lower"):
        """
        Compute the limits based on the loss fraction.

        Parameters
        ----------
        hist : np.ndarray
            2D histogram array.
        bin_edges : np.ndarray
            Array of bin edges.
        loss_fraction : float
            Fraction of events to be lost.
        axis : int, optional
            Axis along which to sum the histogram. Default is 0.
        limit_type : str, optional
            Type of limit ('lower' or 'upper'). Default is 'lower'.

        Returns
        -------
        int
            Bin index where the threshold is reached.
        float
            Bin edge value corresponding to the threshold.
        """
        projection = np.sum(hist, axis=axis)
        cumulative_sum = (
            np.cumsum(projection) if limit_type == "upper" else np.cumsum(projection[::-1])
        )
        total_events = np.sum(projection)
        threshold = (1 - loss_fraction) * total_events
        bin_index = np.searchsorted(cumulative_sum, threshold)
        bin_edge_value = bin_edges[bin_index] if limit_type == "upper" else bin_edges[-bin_index]
        return bin_index, bin_edge_value

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
        num_files = len(self.list_of_files)
        showers_per_file = len(self.simulated) // num_files
        shower_id_triggered_adjusted = self.adjust_shower_ids_across_files(
            self.shower_id_triggered, num_files, showers_per_file
        )

        triggered_energies = self.simulated[shower_id_triggered_adjusted]
        energy_bins = np.logspace(
            np.log10(triggered_energies.min()), np.log10(triggered_energies.max()), 100
        )
        core_distances_all = np.sqrt(self.event_x_core**2 + self.event_y_core**2)
        core_distances_triggered = core_distances_all[shower_id_triggered_adjusted]
        core_bins = np.linspace(core_distances_triggered.min(), core_distances_triggered.max(), 100)

        hist = self._generate_2d_histogram(
            core_distances_triggered, triggered_energies, core_bins, energy_bins
        )
        _, lower_bin_edge_value = self._compute_limits(
            hist, energy_bins, loss_fraction, axis=1, limit_type="lower"
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
        num_files = len(self.list_of_files)
        showers_per_file = len(self.simulated) // num_files
        shower_id_triggered_adjusted = self.adjust_shower_ids_across_files(
            self.shower_id_triggered, num_files, showers_per_file
        )

        triggered_energies = self.simulated[shower_id_triggered_adjusted]
        energy_bins = np.logspace(
            np.log10(triggered_energies.min()), np.log10(triggered_energies.max()), 100
        )
        core_distances_all = np.sqrt(self.event_x_core**2 + self.event_y_core**2)
        core_distances_triggered = core_distances_all[shower_id_triggered_adjusted]
        core_bins = np.linspace(core_distances_triggered.min(), core_distances_triggered.max(), 100)

        hist = self._generate_2d_histogram(
            core_distances_triggered, triggered_energies, core_bins, energy_bins
        )
        _, upper_bin_edge_value = self._compute_limits(
            hist, core_bins, loss_fraction, axis=0, limit_type="upper"
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
        num_files = len(self.list_of_files)
        showers_per_file = len(self.simulated) // num_files
        shower_id_triggered_adjusted = self.adjust_shower_ids_across_files(
            self.shower_id_triggered, num_files, showers_per_file
        )

        azimuth_diff = (self.array_azimuth - self.shower_azimuth) * (np.pi / 180.0)
        sim_altitude_rad = self.shower_sim_altitude * (np.pi / 180.0)
        array_altitude_rad = self.array_altitude * (np.pi / 180.0)

        x_1 = np.cos(azimuth_diff) * np.cos(sim_altitude_rad)
        y_1 = np.sin(azimuth_diff) * np.cos(sim_altitude_rad)
        z_1 = np.sin(sim_altitude_rad)
        x_2 = x_1 * np.sin(array_altitude_rad) - z_1 * np.cos(array_altitude_rad)
        y_2 = y_1
        z_2 = x_1 * np.cos(array_altitude_rad) + z_1 * np.sin(array_altitude_rad)
        off_angles = np.arctan2(np.sqrt(x_2**2 + y_2**2), z_2) * (180.0 / np.pi)

        angle_bins = np.linspace(off_angles.min(), off_angles.max(), 100)
        core_distances_all = np.sqrt(self.event_x_core**2 + self.event_y_core**2)
        core_distances_triggered = core_distances_all[shower_id_triggered_adjusted]
        core_bins = np.linspace(core_distances_triggered.min(), core_distances_triggered.max(), 100)

        hist = self._generate_2d_histogram(
            core_distances_triggered, off_angles, core_bins, angle_bins
        )
        _, upper_bin_edge_value = self._compute_limits(
            hist, angle_bins, loss_fraction, axis=1, limit_type="upper"
        )
        return upper_bin_edge_value * u.deg

    def plot_data(self):
        """Plot the core distances and energies of triggered events."""
        num_files = len(self.list_of_files)
        showers_per_file = len(self.simulated) // num_files
        shower_id_triggered_adjusted = self.adjust_shower_ids_across_files(
            self.shower_id_triggered, num_files, showers_per_file
        )

        core_distances_all = np.sqrt(self.event_x_core**2 + self.event_y_core**2)
        core_distances_triggered = core_distances_all[shower_id_triggered_adjusted]
        triggered_energies = self.simulated[shower_id_triggered_adjusted]

        if len(core_distances_triggered) > 0 and len(triggered_energies) > 0:
            core_bins = np.linspace(
                core_distances_triggered.min(), core_distances_triggered.max(), 100
            )
            energy_bins = np.logspace(
                np.log10(triggered_energies.min()), np.log10(triggered_energies.max()), 100
            )
        else:
            print("Warning: No triggered events found.")
            core_bins = np.linspace(0, 1, 10)
            energy_bins = np.logspace(-1, 1, 10)

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
        adjusted_ids = []
        for file_idx in range(num_files):
            file_offset = file_idx * showers_per_file
            adjusted_ids.append(shower_id_triggered + file_offset)
        return np.concatenate(adjusted_ids)

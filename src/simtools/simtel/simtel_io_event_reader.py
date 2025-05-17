"""
Read a reduced dataset from file.

Allow to filter the events based on the triggered telescopes.
Provide functionality to list events, e.g. through

.. code-block:: console

    from simtools.simtel.simtel_io_event_reader import SimtelIOEventDataReader
    reader = SimtelIOEventDataReader("gamma_diffuse_60deg.hdf5", [1,2,3,4])
    reader.print_event_table()

"""

import logging

import astropy.units as u
import numpy as np
import tables
from astropy.coordinates import AltAz, angular_separation
from ctapipe.coordinates import GroundFrame, TiltedGroundFrame

from simtools.corsika.primary_particle import PrimaryParticle
from simtools.simtel.simtel_io_event_writer import (
    ShowerEventData,
    SimulationFileInfo,
    TriggeredEventData,
)


class SimtelIOEventDataReader:
    """
    Read reduced MC data set from file.

    Calculate some standard derivation like core position in shower coordinates.

    Parameters
    ----------
    event_data_file : str
        Path to the HDF5 file containing the event data.
    telescope_list : list, optional
        List of telescope IDs to filter the events (default is None).
    """

    def __init__(self, event_data_file, telescope_list=None):
        """Initialize SimtelIOEventDataReader with the given event data file."""
        self._logger = logging.getLogger(__name__)
        self.telescope_list = telescope_list

        (
            self.simulation_file_info,
            self.shower_data,
            self.triggered_shower_data,
            self.triggered_data,
        ) = self.read_event_data(event_data_file)
        self._derived_event_data()

    def read_event_data(self, event_data_file):
        """
        Read event data from the reduced MC event data file.

        Parameters
        ----------
        event_data_file : str, Path
            Path to the HDF5 file containing the event data.

        Returns
        -------
        SimulationFileInfo, ShowerEventData, TriggeredEventData
            Simulation file info, event, and triggered event data.
        """
        file_info = SimulationFileInfo()
        event_data = ShowerEventData()
        triggered_event_data = TriggeredEventData()

        with tables.open_file(event_data_file, mode="r") as f:
            file_info = self._read_file_info(f, file_info)
            event_data = self._read_shower_data(f, event_data)
            triggered_data = self._read_triggered_data(f, triggered_event_data)

        triggered_shower, triggered_data = self._reduce_to_triggered_events(
            event_data, triggered_event_data
        )
        return file_info, event_data, triggered_shower, triggered_data

    def _read_file_info(self, file, file_info):
        """Read file information from the HDF5 file."""
        file_data = file.root.data.file_info
        file_info.file_name = file_data.col("file_name")
        file_info.particle_id = file_data.col("particle_id")
        file_info.zenith = file_data.col("zenith")
        file_info.azimuth = file_data.col("azimuth")
        file_info.nsb_level = file_data.col("nsb_level")
        return file_info

    def _read_triggered_data(self, file, triggered_event_data):
        """Read triggered data from the HDF5 file."""
        triggered_data = file.root.data.triggered_data
        triggered_event_data.triggered_id = triggered_data.col("triggered_id")
        triggered_event_data.array_altitudes = triggered_data.col("array_altitudes")
        triggered_event_data.array_azimuths = triggered_data.col("array_azimuths")

        telescope_indices = triggered_data.col("telescope_list_index")
        telescope_list_array = file.root.data.trigger_telescope_list_list
        triggered_event_data.trigger_telescope_list_list = []
        for index in telescope_indices:
            if index < telescope_list_array.nrows:
                triggered_event_data.trigger_telescope_list_list.append(telescope_list_array[index])
            else:
                self._logger.warning(f"Invalid telescope list index: {index}")
                triggered_event_data.trigger_telescope_list_list.append(
                    np.array([], dtype=np.int16)
                )

        return triggered_event_data

    @staticmethod
    def _read_shower_data(file, event_data):
        """Read shower data from the HDF5 file."""
        reduced_data = file.root.data.reduced_data
        event_data.simulated_energy = reduced_data.col("simulated_energy")
        event_data.x_core = reduced_data.col("x_core")
        event_data.y_core = reduced_data.col("y_core")
        event_data.shower_azimuth = reduced_data.col("shower_azimuth")
        event_data.shower_altitude = reduced_data.col("shower_altitude")
        event_data.shower_id = reduced_data.col("shower_id")
        event_data.area_weight = reduced_data.col("area_weight")
        return event_data

    def _reduce_to_triggered_events(self, event_data, triggered_data):
        """
        Reduce event data to triggered events only. Apply filter on telescope list if specified.

        Parameters
        ----------
        event_data : ShowerEventData
            Event data.
        triggered_data : TriggeredEventData
            Triggered event data.

        Returns
        -------
        ShowerEventData, TriggeredEventData
            Filtered event data and triggered event data
        """
        filtered_shower_ids, triggered_indices = self._get_mask_triggered_telescopes(
            self.telescope_list,
            triggered_data.triggered_id,
            triggered_data.trigger_telescope_list_list,
        )
        filtered_event_data = ShowerEventData(
            x_core=event_data.x_core[filtered_shower_ids],
            y_core=event_data.y_core[filtered_shower_ids],
            simulated_energy=event_data.simulated_energy[filtered_shower_ids],
            shower_azimuth=event_data.shower_azimuth[filtered_shower_ids],
            shower_altitude=event_data.shower_altitude[filtered_shower_ids],
        )

        filtered_telescope_list = [
            triggered_data.trigger_telescope_list_list[i] for i in triggered_indices
        ]

        filtered_triggered_data = TriggeredEventData(
            array_azimuths=triggered_data.array_azimuths[triggered_indices],
            array_altitudes=triggered_data.array_altitudes[triggered_indices],
            trigger_telescope_list_list=filtered_telescope_list,
        )
        self._logger.info(
            f"Events reduced to triggered events: {len(filtered_event_data.simulated_energy)}"
        )
        return filtered_event_data, filtered_triggered_data

    def _get_mask_triggered_telescopes(
        self, telescope_list, triggered_id, trigger_telescope_list_list
    ):
        """
        Return indices of events that triggered the specified telescopes.

        Parameters
        ----------
        telescope_list : list
            List of telescope IDs to filter the events
        triggered_id : np.ndarray
            Array of event IDs that triggered.
        trigger_telescope_list_list : list
            List of triggered telescopes for each event.

        Returns
        -------
        tuple
            Filtered triggered IDs and indices.
        """
        triggered_indices = np.arange(len(triggered_id))
        if telescope_list:
            mask = np.array(
                [
                    any(tel in event for tel in telescope_list)
                    for event in trigger_telescope_list_list
                ]
            )
            triggered_indices = triggered_indices[mask]
            return triggered_id[mask], triggered_indices
        return triggered_id, triggered_indices

    def _derived_event_data(self):
        """Calculate core positions in shower coordinates and angular distances."""
        for event_data in (self.shower_data, self.triggered_shower_data):
            event_data.x_core_shower, event_data.y_core_shower = (
                self._transform_to_shower_coordinates(
                    event_data.x_core,
                    event_data.y_core,
                    event_data.shower_azimuth,
                    event_data.shower_altitude,
                )
            )
            event_data.core_distance_shower = np.sqrt(
                event_data.x_core_shower**2 + event_data.y_core_shower**2
            )

        self.triggered_data.angular_distance = (
            angular_separation(
                self.triggered_shower_data.shower_azimuth,
                self.triggered_shower_data.shower_altitude,
                self.triggered_data.array_azimuths,
                self.triggered_data.array_altitudes,
            )
            * 180
            / np.pi
        )

    def _transform_to_shower_coordinates(self, x_core, y_core, shower_azimuth, shower_altitude):
        """
        Transform core positions from ground coordinates to shower coordinates.

        Parameters
        ----------
        x_core : np.ndarray
            Core x positions in ground coordinates.
        y_core : np.ndarray
            Core y positions in ground coordinates.
        shower_azimuth : np.ndarray
            Shower azimuth angles.
        shower_altitude : np.ndarray
            Shower altitude angles.

        Returns
        -------
        tuple
            Core positions in shower coordinates (x, y).
        """
        pointing = AltAz(az=shower_azimuth * u.rad, alt=shower_altitude * u.rad)
        ground = GroundFrame(x=x_core * u.m, y=y_core * u.m, z=0 * u.m)
        shower_frame = ground.transform_to(TiltedGroundFrame(pointing_direction=pointing))

        return shower_frame.x.value, shower_frame.y.value

    def print_dataset_information(self, n_events=1):
        """Print information about the datasets."""

        def print_event_data(data, name):
            """Print event data."""
            print(f"{name}: {data[:n_events]}")

        print_event_data(self.triggered_shower_data.simulated_energy, "Simulated energy (TeV)")
        print_event_data(self.triggered_shower_data.x_core, "Core x (m)")
        print_event_data(self.triggered_shower_data.y_core, "Core y (m)")
        print_event_data(self.triggered_shower_data.shower_azimuth, "Shower azimuth (rad)")
        print_event_data(self.triggered_shower_data.shower_altitude, "Shower altitude (rad)")
        print_event_data(self.triggered_shower_data.x_core_shower, "Core x shower (m)")
        print_event_data(self.triggered_shower_data.y_core_shower, "Core y shower (m)")
        print_event_data(
            self.triggered_shower_data.core_distance_shower, "Core distance shower (m)"
        )
        print_event_data(self.triggered_data.array_azimuths, "Array azimuth (rad)")
        print_event_data(self.triggered_data.array_altitudes, "Array altitude (rad)")
        print_event_data(self.triggered_data.trigger_telescope_list_list, "Triggered telescopes")
        print_event_data(
            self.triggered_data.angular_distance, "Angular distance to pointing direction (deg)"
        )
        print("")

    def print_event_table(self):
        """Print event table."""
        print(
            f"{'Counter':<10} {'Simulated Energy (TeV)':<20} {'Triggered Telescopes':<20} "
            f"{'Core distance shower (m)':<20}"
        )

        for i, telescope_list in enumerate(self.triggered_data.trigger_telescope_list_list):
            print(
                f"{i:<10} {self.triggered_shower_data.simulated_energy[i]:<20.3f}"
                f"{telescope_list} "
                f"{self.triggered_shower_data.core_distance_shower[i]:<20.3f}"
            )
        print("")

    def get_reduced_simulation_file_info(self):
        """
        Return reduced simulation file info assuming single-valued parameters.

        Applies rounding and uniqueness checks to extract representative values
        for zenith, azimuth, and NSB level. Assumes all files share identical
        simulation parameters except for file names. Returns particle name instead
        of ID.

        Logs a warning if multiple unique values are found.

        Returns
        -------
        dict
            Dictionary containing the reduced simulation file info.
        """
        particle_id = np.unique(self.simulation_file_info.particle_id)
        zenith = np.unique(np.round(self.simulation_file_info.zenith, decimals=2))
        azimuth = np.unique(np.round(self.simulation_file_info.azimuth, decimals=2))
        nsb = np.unique(np.round(self.simulation_file_info.nsb_level, decimals=2))
        # TODO - check if this should be an error or an warning
        if any(len(arr) > 1 for arr in (particle_id, zenith, azimuth, nsb)):
            self._logger.warning("Simulation file info has non-unique values.")

        return {
            "primary_particle": PrimaryParticle(
                particle_id_type="corsika7_id",
                particle_id=particle_id[0],
            ).name,
            "zenith": zenith[0],
            "azimuth": azimuth[0],
            "nsb_level": nsb[0],
        }

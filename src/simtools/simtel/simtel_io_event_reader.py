"""
Read reduced datasets from FITS tables.

Allow to filter the events based on the triggered telescopes.
Provide functionality to list events, e.g. through

.. code-block:: console

    from simtools.simtel.simtel_io_event_reader import SimtelIOEventDataReader
    reader = SimtelIOEventDataReader("gamma_diffuse_60deg.hdf5", [1,2,3,4])
    reader.print_event_table()

"""

import logging
from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, angular_separation
from astropy.table import Table
from ctapipe.coordinates import GroundFrame, TiltedGroundFrame

from simtools.corsika.primary_particle import PrimaryParticle


@dataclass
class ShowerEventData:
    """Container for shower event data."""

    shower_id: list[np.uint32] = field(default_factory=list)
    event_id: list[np.uint32] = field(default_factory=list)
    file_id: list[np.uint32] = field(default_factory=list)
    simulated_energy: list[np.float64] = field(default_factory=list)
    x_core: list[np.float64] = field(default_factory=list)
    y_core: list[np.float64] = field(default_factory=list)
    shower_azimuth: list[np.float64] = field(default_factory=list)
    shower_altitude: list[np.float64] = field(default_factory=list)
    area_weight: list[np.float64] = field(default_factory=list)
    x_core_shower: list[np.float64] = field(default_factory=list)
    y_core_shower: list[np.float64] = field(default_factory=list)
    core_distance_shower: list[np.float64] = field(default_factory=list)


@dataclass
class TriggeredEventData:
    """Container for triggered event data."""

    shower_id: list[np.uint32] = field(default_factory=list)
    event_id: list[np.uint32] = field(default_factory=list)
    file_id: list[np.uint32] = field(default_factory=list)
    array_altitude: list[float] = field(default_factory=list)
    array_azimuth: list[float] = field(default_factory=list)
    telescope_list: list[np.ndarray] = field(default_factory=list)
    angular_distance: list[float] = field(default_factory=list)


class SimtelIOEventDataReader:
    """Read reduced MC data set stored in astropy tables."""

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

    def _table_to_shower_data(self, table):
        """
        Convert tables with shower event data and add derived quantities.

        Parameters
        ----------
        table : astropy.table.Table
            Table "SHOWERS"

        Returns
        -------
        ShowerEventData
            An instance of ShowerEventData with the data from the table.
        """
        shower_data = ShowerEventData()

        for col in table.colnames:
            setattr(shower_data, col, np.array(table[col].data))
            if table[col].unit:
                setattr(shower_data, f"{col}_unit", table[col].unit)

        shower_data.x_core_shower, shower_data.y_core_shower = (
            self._transform_to_shower_coordinates(
                shower_data.x_core,
                shower_data.y_core,
                shower_data.shower_azimuth,
                shower_data.shower_altitude,
            )
        )
        shower_data.core_distance_shower = np.sqrt(
            shower_data.x_core_shower**2 + shower_data.y_core_shower**2
        )

        return shower_data

    def _table_to_triggered_data(self, table):
        """
        Convert table with triggered event data.

        Parameters
        ----------
        table : astropy.table.Table
            Table "TRIGGERS"

        Returns
        -------
        TriggeredEventData
            An instance of TriggeredEventData with the data from the table.
        """
        triggered_data = TriggeredEventData()
        for col in table.colnames:
            if col == "telescope_list":
                arrays = [
                    np.array(list(map(int, tel_list.split(","))), dtype=np.int16)
                    for tel_list in table[col]
                ]
                triggered_data.telescope_list = arrays
            else:
                data = np.array(table[col].data)
                setattr(triggered_data, col, data)
                if table[col].unit:
                    setattr(triggered_data, f"{col}_unit", table[col].unit)
        return triggered_data

    def _get_triggered_shower_data(self, shower_data, trigger_table):
        """Get shower data corresponding to triggered events."""
        triggered_shower = ShowerEventData()

        matched_indices = []
        for tr_shower_id, tr_event_id, tr_file_id in zip(
            trigger_table["shower_id"], trigger_table["event_id"], trigger_table["file_id"]
        ):
            mask = (
                (shower_data.shower_id == tr_shower_id)
                & (shower_data.event_id == tr_event_id)
                & (shower_data.file_id == tr_file_id)
            )
            matched_idx = np.where(mask)[0]
            if len(matched_idx) == 1:
                matched_indices.append(matched_idx[0])
            else:
                self._logger.warning(
                    f"Found {len(matched_idx)} matches for shower {tr_shower_id}"
                    f" event {tr_event_id} file {tr_file_id}"
                )

        for attr in vars(shower_data):
            if not attr.endswith("_unit"):
                value = getattr(shower_data, attr)
                if isinstance(value, list | np.ndarray):
                    setattr(triggered_shower, attr, np.array(value)[matched_indices])

        return triggered_shower

    def read_event_data(self, event_data_file):
        """Read event data from FITS file."""
        shower_table = Table.read(event_data_file, hdu="SHOWERS")
        trigger_table = Table.read(event_data_file, hdu="TRIGGERS")
        file_info_table = Table.read(event_data_file, hdu="FILE_INFO")

        shower_data = self._table_to_shower_data(shower_table)
        triggered_data = self._table_to_triggered_data(trigger_table)
        triggered_shower = self._get_triggered_shower_data(shower_data, trigger_table)

        triggered_data.angular_distance = (
            angular_separation(
                triggered_shower.shower_azimuth * u.rad,
                triggered_shower.shower_altitude * u.rad,
                triggered_data.array_azimuth * u.rad,
                triggered_data.array_altitude * u.rad,
            )
            .to(u.deg)
            .value
        )

        triggered_data = (
            self._filter_by_telescopes(triggered_data) if self.telescope_list else triggered_data
        )

        return file_info_table, shower_data, triggered_shower, triggered_data

    def _filter_by_telescopes(self, triggered_data):
        """Filter triggered data by the specified telescope list."""
        mask = np.array(
            [
                any(tel in event for tel in self.telescope_list)
                for event in triggered_data.telescope_list
            ]
        )
        filtered_triggered_data = TriggeredEventData(
            array_altitude=triggered_data.array_altitude[mask],
            array_azimuth=triggered_data.array_azimuth[mask],
            telescope_list=[triggered_data.telescope_list[i] for i in np.arange(len(mask))[mask]],
            angular_distance=triggered_data.angular_distance[mask],
        )
        self._logger.info(
            f"Events reduced to triggered events: {len(filtered_triggered_data.array_altitude)}"
        )
        return filtered_triggered_data

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
        ground = GroundFrame(x=x_core * u.m, y=y_core * u.m, z=np.zeros_like(x_core) * u.m)
        shower_frame = ground.transform_to(
            TiltedGroundFrame(
                pointing_direction=AltAz(az=shower_azimuth * u.rad, alt=shower_altitude * u.rad)
            )
        )
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
        print_event_data(self.triggered_data.array_azimuth, "Array azimuth (rad)")
        print_event_data(self.triggered_data.array_altitude, "Array altitude (rad)")
        print_event_data(self.triggered_data.telescope_list, "Triggered telescopes")
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

        for i, telescope_list in enumerate(self.triggered_data.telescope_list):
            print(
                f"{i:<10} {self.triggered_shower_data.simulated_energy[i]:<20.3f}"
                f"{telescope_list} "
                f"{self.triggered_shower_data.core_distance_shower[i]:<20.3f}"
            )
        print("")

    def get_reduced_simulation_file_info(self):
        """
        Return reduced simulation file info assuming single-valued parameters.

        Applies rounding and uniqueness functions extract representative values
        for zenith, azimuth, and NSB level. Assumes all files share identical
        simulation parameters except for file names. Returns particle name instead
        of ID.

        Logs a warning if multiple unique values are found.

        Returns
        -------
        dict
            Dictionary containing the reduced simulation file info.
        """
        particle_id = np.unique(self.simulation_file_info["particle_id"].data)
        keys = ["zenith", "azimuth", "nsb_level"]
        float_arrays = {}
        for key in keys:
            float_arrays[key] = np.unique(np.round(self.simulation_file_info[key].data, decimals=2))

        if any(len(arr) > 1 for arr in (particle_id, *(float_arrays[key] for key in keys))):
            self._logger.warning("Simulation file info has non-unique values.")

        reduced_info = {
            "primary_particle": PrimaryParticle(
                particle_id_type="corsika7_id",
                particle_id=int(particle_id[0]),
            ).name,
        }

        for key in keys:
            value = float(float_arrays[key][0])
            if self.simulation_file_info[key].unit is not None:
                value = value * self.simulation_file_info[key].unit
            reduced_info[key] = value

        return reduced_info

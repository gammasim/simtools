"""Read reduced datasets in form of astropy tables from file."""

import logging
from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
from astropy.coordinates import angular_separation

from simtools.corsika.primary_particle import PrimaryParticle
from simtools.io import table_handler
from simtools.utils.geometry import solid_angle, transform_ground_to_shower_coordinates


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
    angular_distance: list[float] = field(default_factory=list)


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


class EventDataReader:
    """Read reduced MC data set stored in astropy tables."""

    def __init__(self, event_data_file, telescope_list=None):
        """Initialize EventDataReader."""
        self._logger = logging.getLogger(__name__)
        self.telescope_list = telescope_list

        self.data_sets = self.read_table_list(event_data_file)
        self.reduced_file_info = None

    def read_table_list(self, event_data_file):
        """
        Read available tables from the event data file.

        Rearrange dictionary with tables names into a list of dictionaries
        under the assumption that the file contains the tables "SHOWERS",
        "TRIGGERS", and "FILE_INFO". Note that not all tables need to be present.

        Parameters
        ----------
        event_data_file : str
            Path to the event data file.

        Returns
        -------
        list
            List of dictionaries containing the data from the tables.
        """
        dataset_dict = table_handler.read_table_list(
            event_data_file,
            ["SHOWERS", "TRIGGERS", "FILE_INFO"],
            include_indexed_tables=True,
        )

        data_sets = []
        try:
            sorted_indices = sorted(
                range(len(dataset_dict["SHOWERS"])),
                key=lambda i: int(dataset_dict["SHOWERS"][i].split("_")[-1]),
            )
        except (ValueError, AttributeError):
            sorted_indices = [0]  # Handle the case where the key is only "SHOWERS"
        for i in sorted_indices:
            entry = {
                "SHOWERS": dataset_dict["SHOWERS"][i],
                "FILE_INFO": dataset_dict["FILE_INFO"][i],
            }
            if i < len(dataset_dict["TRIGGERS"]) and dataset_dict["TRIGGERS"][i]:
                entry["TRIGGERS"] = dataset_dict["TRIGGERS"][i]
            data_sets.append(entry)

        return data_sets

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

        shower_data.x_core_shower, shower_data.y_core_shower, _ = (
            transform_ground_to_shower_coordinates(
                shower_data.x_core,
                shower_data.y_core,
                0.0,
                shower_data.shower_azimuth,
                shower_data.shower_altitude,
            )
        )
        shower_data.core_distance_shower = np.hypot(
            shower_data.x_core_shower, shower_data.y_core_shower
        )
        shower_data.angular_distance = (
            angular_separation(
                shower_data.shower_azimuth * u.deg,
                shower_data.shower_altitude * u.deg,
                self.reduced_file_info["azimuth"],
                (90.0 * u.deg - self.reduced_file_info["zenith"]),
            )
            .to(u.deg)
            .value
        )

        return shower_data

    def _table_to_triggered_data(self, table):
        """
        Convert table with triggered event data.

        Converts telescope lists from comma-separated string to numpy array.

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
                    np.array(list(map(str, tel_list.split(","))), dtype=np.str_)
                    for tel_list in table[col]
                ]
                triggered_data.telescope_list = arrays
            else:
                data = np.array(table[col].data)
                setattr(triggered_data, col, data)
                if table[col].unit:
                    setattr(triggered_data, f"{col}_unit", table[col].unit)
        return triggered_data

    def _get_triggered_shower_data(
        self, shower_data, triggered_file_id, triggered_event_id, triggered_shower_id
    ):
        """
        Get shower data corresponding to triggered events.

        Matches triggered events with showers based on shower_id, event_id, and file_id.

        Parameters
        ----------
        shower_data : ShowerEventData
            The shower data containing all showers.
        triggered_file_id : list
            List of file IDs for triggered events.
        triggered_event_id : list
            List of event IDs for triggered events.
        triggered_shower_id : list
            List of shower IDs for triggered events.

        Returns
        -------
        ShowerEventData
            An instance of ShowerEventData containing only the triggered showers.

        """
        triggered_shower = ShowerEventData()

        matched_indices = []
        for tr_shower_id, tr_event_id, tr_file_id in zip(
            triggered_shower_id, triggered_event_id, triggered_file_id
        ):
            mask = (
                (shower_data.shower_id == tr_shower_id)
                & (shower_data.event_id == tr_event_id)
                & (shower_data.file_id == tr_file_id)
            )
            matched_idx = np.nonzero(mask)[0]
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

    def read_event_data(self, event_data_file, table_name_map=None):
        """
        Read event data and file info tables from file and apply transformations.

        Allows to map tables names to their actual names in the file
        (e.g., "SHOWER" to "SHOWER_1").

        Parameters
        ----------
        event_data_file : str
            Path to the event data file.
        table_name_map : dict, optional
            Mapping of table names to their actual names in the file.
            Defaults to using the standard names "SHOWERS", "TRIGGERS", and "FILE_INFO".

        Returns
        -------
        tuple
            A tuple with file info table, shower, triggered shower, and triggered event data.
        """

        def get_name(k):
            return k if table_name_map is None else table_name_map.get(k)

        table_names = [
            name for k in ("SHOWERS", "TRIGGERS", "FILE_INFO") if (name := get_name(k)) is not None
        ]
        tables = table_handler.read_tables(event_data_file, table_names=table_names)
        self.reduced_file_info = self.get_reduced_simulation_file_info(
            tables[get_name("FILE_INFO")]
        )

        shower_data = self._table_to_shower_data(tables[get_name("SHOWERS")])
        if tables.get(get_name("TRIGGERS")) is None:
            self._logger.info("No triggered event data found in the file.")
            return tables[get_name("FILE_INFO")], shower_data, None, None

        triggered_data = self._table_to_triggered_data(tables[get_name("TRIGGERS")])
        triggered_shower = self._get_triggered_shower_data(
            shower_data,
            tables[get_name("TRIGGERS")]["file_id"],
            tables[get_name("TRIGGERS")]["event_id"],
            tables[get_name("TRIGGERS")]["shower_id"],
        )

        triggered_data.angular_distance = (
            angular_separation(
                triggered_shower.shower_azimuth * u.deg,
                triggered_shower.shower_altitude * u.deg,
                triggered_data.array_azimuth * u.deg,
                triggered_data.array_altitude * u.deg,
            )
            .to(u.deg)
            .value
        )

        if self.telescope_list:
            triggered_data, triggered_shower = self._filter_by_telescopes(
                triggered_data, triggered_shower
            )

        self._logger.info(f"Number of triggered events: {len(triggered_data.array_altitude)}")

        return (
            tables[get_name("FILE_INFO")],
            shower_data,
            triggered_shower,
            triggered_data,
        )

    def _filter_by_telescopes(self, triggered_data, triggered_shower):
        """Filter trigger data and triggered shower data by the specified telescope list."""
        mask = np.array(
            [
                any(tel in event for tel in self.telescope_list)
                for event in triggered_data.telescope_list
            ]
        )
        filtered_triggered_data = TriggeredEventData(
            shower_id=triggered_data.shower_id[mask],
            event_id=triggered_data.event_id[mask],
            file_id=triggered_data.file_id[mask],
            array_altitude=triggered_data.array_altitude[mask],
            array_azimuth=triggered_data.array_azimuth[mask],
            telescope_list=[triggered_data.telescope_list[i] for i in np.arange(len(mask))[mask]],
            angular_distance=triggered_data.angular_distance[mask],
        )
        filtered_triggered_shower_data = self._get_triggered_shower_data(
            triggered_shower,
            filtered_triggered_data.file_id,
            filtered_triggered_data.event_id,
            filtered_triggered_data.shower_id,
        )

        return filtered_triggered_data, filtered_triggered_shower_data

    def get_reduced_simulation_file_info(self, simulation_file_info):
        """
        Return reduced simulation file info assuming single-valued parameters.

        Applies rounding and uniqueness functions extract representative values
        for zenith, azimuth, and NSB level. Assumes all files share identical
        simulation parameters except for file names. Returns particle name instead
        of ID.

        Logs a warning if multiple unique values are found.

        Parameters
        ----------
        simulation_file_info : astropy.table.Table
            Dictionary containing simulation file info.

        Returns
        -------
        dict
            Dictionary containing the reduced simulation file info.
        """
        particle_id = np.unique(simulation_file_info["particle_id"].data)
        keys = [
            "zenith",
            "azimuth",
            "nsb_level",
            "energy_min",
            "energy_max",
            "viewcone_min",
            "viewcone_max",
            "core_scatter_min",
            "core_scatter_max",
        ]
        float_arrays = {}
        for key in keys:
            if key == "energy_min":
                float_arrays[key] = np.unique(np.round(simulation_file_info[key].data, decimals=3))
            else:
                float_arrays[key] = np.unique(np.round(simulation_file_info[key].data, decimals=2))

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
            if simulation_file_info[key].unit is not None:
                value = value * simulation_file_info[key].unit
            reduced_info[key] = value

        reduced_info["solid_angle"] = solid_angle(
            angle_min=reduced_info.get("viewcone_min", 0.0 * u.rad),
            angle_max=reduced_info.get("viewcone_max", 0.0 * u.rad),
        )
        reduced_info["scatter_area"] = self.scatter_area(
            core_scatter_min=reduced_info.get("core_scatter_min", 0.0 * u.m),
            core_scatter_max=reduced_info.get("core_scatter_max", 0.0 * u.m),
        )

        return reduced_info

    def scatter_area(self, core_scatter_min, core_scatter_max):
        """
        Calculate the scatter area of the core.

        Parameters
        ----------
        core_scatter_min : astropy.units.Quantity
            Minimum core scatter radius.
        core_scatter_max : astropy.units.Quantity
            Maximum core scatter radius.

        Returns
        -------
        astropy.units.Quantity
            Scatter area.
        """
        return np.pi * (core_scatter_max**2 - core_scatter_min**2)

"""Read reduced datasets in form of astropy tables from file."""

import logging
from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
from astropy.coordinates import angular_separation

from simtools.corsika.primary_particle import PrimaryParticle
from simtools.io import table_handler
from simtools.utils.geometry import project_ground_to_corsika_shower_coordinates, solid_angle


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

    _particle_name_cache = {}
    _required_shower_columns = [
        "shower_id",
        "event_id",
        "file_id",
        "simulated_energy",
        "x_core",
        "y_core",
        "shower_azimuth",
        "shower_altitude",
        "area_weight",
    ]
    _required_trigger_columns = [
        "shower_id",
        "event_id",
        "file_id",
        "array_altitude",
        "array_azimuth",
        "telescope_list",
    ]
    _required_file_info_columns = [
        "particle_id",
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

    def __init__(self, event_data_file, telescope_list=None):
        """Initialize EventDataReader."""
        self._logger = logging.getLogger(__name__)
        self.telescope_list = telescope_list

        self._validate_hdf5_event_data_file(event_data_file)
        self.data_sets = self.read_table_list(event_data_file)
        self.reduced_file_info = None

    @staticmethod
    def _validate_hdf5_event_data_file(event_data_file):
        """Validate that reduced event data are stored in HDF5 format."""
        if table_handler.read_table_file_type([event_data_file]) != "HDF5":
            raise ValueError(
                f"Unsupported reduced event data format for '{event_data_file}'. "
                "Only HDF5 files with suffix '.hdf5' or '.h5' are supported."
            )

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
            project_ground_to_corsika_shower_coordinates(
                shower_data.x_core,
                shower_data.y_core,
                0.0,
                np.deg2rad(shower_data.shower_azimuth),
                np.deg2rad(shower_data.shower_altitude),
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

        key_dtype = np.dtype(
            [("shower_id", np.int64), ("event_id", np.int64), ("file_id", np.int64)]
        )

        def build_keys(shower_ids, event_ids, file_ids):
            keys = np.empty(len(shower_ids), dtype=key_dtype)
            keys["shower_id"] = np.asarray(shower_ids, dtype=np.int64)
            keys["event_id"] = np.asarray(event_ids, dtype=np.int64)
            keys["file_id"] = np.asarray(file_ids, dtype=np.int64)
            return keys

        shower_keys = build_keys(
            shower_data.shower_id,
            shower_data.event_id,
            shower_data.file_id,
        )
        trigger_keys = build_keys(
            triggered_shower_id,
            triggered_event_id,
            triggered_file_id,
        )

        order = np.argsort(shower_keys, kind="stable")
        sorted_shower_keys = shower_keys[order]
        positions = np.searchsorted(sorted_shower_keys, trigger_keys)

        found = positions < len(sorted_shower_keys)
        if np.any(found):
            found_indices = np.flatnonzero(found)
            found[found_indices] = (
                sorted_shower_keys[positions[found_indices]] == trigger_keys[found_indices]
            )

        duplicate_sorted = np.zeros(len(sorted_shower_keys), dtype=np.bool_)
        if len(sorted_shower_keys) > 1:
            equal_neighbors = sorted_shower_keys[1:] == sorted_shower_keys[:-1]
            duplicate_sorted[:-1] |= equal_neighbors
            duplicate_sorted[1:] |= equal_neighbors

        duplicate_match = np.zeros(len(trigger_keys), dtype=np.bool_)
        if np.any(found):
            found_indices = np.flatnonzero(found)
            duplicate_match[found_indices] = duplicate_sorted[positions[found_indices]]

        for trigger_index in np.flatnonzero(~found | duplicate_match):
            if duplicate_match[trigger_index]:
                self._logger.warning(
                    f"Found multiple matches for shower {triggered_shower_id[trigger_index]}"
                    f" event {triggered_event_id[trigger_index]}"
                    f" file {triggered_file_id[trigger_index]}"
                )
            else:
                self._logger.warning(
                    f"Found 0 matches for shower {triggered_shower_id[trigger_index]}"
                    f" event {triggered_event_id[trigger_index]}"
                    f" file {triggered_file_id[trigger_index]}"
                )

        valid_matches = found & ~duplicate_match
        matched_indices = order[positions[valid_matches]]

        for attr in vars(shower_data):
            if not attr.endswith("_unit"):
                value = getattr(shower_data, attr)
                if isinstance(value, list | np.ndarray):
                    setattr(triggered_shower, attr, np.asarray(value)[matched_indices])

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
        self._validate_hdf5_event_data_file(event_data_file)

        def get_name(k):
            return k if table_name_map is None else table_name_map.get(k)

        table_names = [
            name for k in ("SHOWERS", "TRIGGERS", "FILE_INFO") if (name := get_name(k)) is not None
        ]

        table_columns = {
            get_name("SHOWERS"): self._required_shower_columns,
        }
        triggers_name = get_name("TRIGGERS")
        if triggers_name is not None:
            table_columns[triggers_name] = self._required_trigger_columns

        try:
            tables = table_handler.read_tables(
                event_data_file,
                table_names=table_names,
                file_type="HDF5",
                table_columns=table_columns,
            )
        except KeyError as exc:
            raise ValueError(
                f"Reduced event data file '{event_data_file}' is missing a required "
                f"table or column: {exc}."
            ) from exc
        self._validate_event_data_tables(tables, event_data_file, table_names, get_name)
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
            triggered_data, triggered_shower = self.filter_by_telescopes(
                triggered_data, triggered_shower
            )

        self._logger.info(f"Number of triggered events: {len(triggered_data.array_altitude)}")

        return (
            tables[get_name("FILE_INFO")],
            shower_data,
            triggered_shower,
            triggered_data,
        )

    def _validate_event_data_tables(self, tables, event_data_file, table_names, get_name):
        """Validate reduced event-data table presence, row counts, and numeric dtypes."""
        empty_tables = [name for name in table_names if len(tables[name]) == 0]
        if empty_tables:
            raise ValueError(
                f"Reduced event data file '{event_data_file}' has empty required "
                f"table(s): {', '.join(empty_tables)}."
            )

        for table_name, columns in (
            (get_name("SHOWERS"), self._required_shower_columns),
            (get_name("TRIGGERS"), self._required_trigger_columns[:-1]),
        ):
            if table_name is None:
                continue
            invalid_columns = [
                col
                for col in columns
                if np.asarray(tables[table_name][col].data).dtype.kind not in "uif"
            ]
            if invalid_columns:
                raise ValueError(
                    f"Reduced event data file '{event_data_file}' table '{table_name}' "
                    "has non-numeric dtype for required numeric column(s): "
                    f"{', '.join(invalid_columns)}."
                )

        file_info_name = get_name("FILE_INFO")
        if file_info_name is not None:
            missing_columns = [
                column
                for column in self._required_file_info_columns
                if column not in tables[file_info_name].colnames
            ]
            if missing_columns:
                raise ValueError(
                    f"Reduced event data file '{event_data_file}' table '{file_info_name}' "
                    "is missing required column(s): "
                    f"{', '.join(missing_columns)}."
                )

    def filter_by_telescopes(self, triggered_data, triggered_shower, telescope_list=None):
        """
        Filter trigger and shower data by an explicit or reader telescope list.

        Parameters
        ----------
        triggered_data : TriggeredEventData
            Triggered event data to filter.
        triggered_shower : ShowerEventData
            Shower data corresponding to the triggered events.
        telescope_list : list, optional
            List of telescopes to filter by. If None, uses the reader's telescope list.

        Returns
        -------
        tuple
            A tuple containing the filtered triggered event data and the corresponding
            triggered shower data.
        """
        telescope_set = set(self.telescope_list if telescope_list is None else telescope_list)
        mask = np.fromiter(
            (not telescope_set.isdisjoint(event) for event in triggered_data.telescope_list),
            dtype=np.bool_,
            count=len(triggered_data.telescope_list),
        )
        selected_indices = np.flatnonzero(mask)

        filtered_triggered_data = TriggeredEventData(
            shower_id=triggered_data.shower_id[mask],
            event_id=triggered_data.event_id[mask],
            file_id=triggered_data.file_id[mask],
            array_altitude=triggered_data.array_altitude[mask],
            array_azimuth=triggered_data.array_azimuth[mask],
            telescope_list=[triggered_data.telescope_list[i] for i in selected_indices],
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
        particle_id = np.unique(
            self._convert_numeric_column(
                simulation_file_info["particle_id"].data,
                key="particle_id",
                dtype=np.int64,
            )
        )
        keys = [
            "zenith",
            "azimuth",
            "nsb_level",
            "spectral_index",
            "energy_min",
            "energy_max",
            "viewcone_min",
            "viewcone_max",
            "core_scatter_min",
            "core_scatter_max",
        ]
        float_arrays = {}
        for key in keys:
            if key not in simulation_file_info.colnames:
                continue
            column_values = self._convert_numeric_column(
                simulation_file_info[key].data,
                key=key,
                dtype=np.float64,
            )
            if key == "energy_min":
                float_arrays[key] = np.unique(np.round(column_values, decimals=3))
            else:
                float_arrays[key] = np.unique(np.round(column_values, decimals=2))

        if any(len(arr) > 1 for arr in (particle_id, *float_arrays.values())):
            self._logger.warning("Simulation file info has non-unique values.")

        primary_particle_id = int(particle_id[0])
        if primary_particle_id not in self._particle_name_cache:
            self._particle_name_cache[primary_particle_id] = PrimaryParticle(
                particle_id_type="corsika7_id",
                particle_id=primary_particle_id,
            ).name

        reduced_info = {
            "primary_particle": self._particle_name_cache[primary_particle_id],
        }

        for key in keys:
            if key not in float_arrays:
                continue
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

    def _convert_numeric_column(self, values, key, dtype):
        """Convert table column values to a numeric numpy array."""
        array_values = np.asarray(values)

        if array_values.dtype.kind == "O":
            array_values = np.array(
                [
                    value.decode("utf-8") if isinstance(value, bytes | bytearray) else value
                    for value in array_values
                ]
            )

        try:
            if array_values.dtype.kind in ("S", "U"):
                return np.char.strip(array_values.astype("U")).astype(dtype)
            return array_values.astype(dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Unable to convert FILE_INFO column '{key}' to numeric values."
            ) from exc

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

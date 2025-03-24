"""Generate a reduced dataset from given simulation event list and save the output to file."""

import logging
from dataclasses import dataclass, field

import numpy as np
import tables
from eventio import EventIOFile
from eventio.simtel import (
    ArrayEvent,
    MCEvent,
    MCRunHeader,
    MCShower,
    TrackingPosition,
    TriggerInformation,
)

DEFAULT_FILTERS = tables.Filters(complevel=5, complib="zlib", shuffle=True, bitshuffle=False)


@dataclass
class ShowerEventData:
    """Shower event data."""

    simulated_energy: np.ndarray = field(default_factory=lambda: np.array([]))
    x_core: np.ndarray = field(default_factory=lambda: np.array([]))
    y_core: np.ndarray = field(default_factory=lambda: np.array([]))
    shower_azimuth: np.ndarray = field(default_factory=lambda: np.array([]))
    shower_altitude: np.ndarray = field(default_factory=lambda: np.array([]))
    shower_id: np.ndarray = field(default_factory=lambda: np.array([]))
    area_weight: np.ndarray = field(default_factory=lambda: np.array([]))

    x_core_shower: np.ndarray = field(default_factory=lambda: np.array([]))
    y_core_shower: np.ndarray = field(default_factory=lambda: np.array([]))
    core_distance_shower: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class TriggeredEventData:
    """Triggered event data."""

    triggered_id: np.ndarray = field(default_factory=lambda: np.array([]))
    triggered_energy: np.ndarray = field(default_factory=lambda: np.array([]))
    array_altitudes: np.ndarray = field(default_factory=lambda: np.array([]))
    array_azimuths: np.ndarray = field(default_factory=lambda: np.array([]))
    trigger_telescope_list_list: list = field(default_factory=list)
    angular_distance: np.ndarray = field(default_factory=lambda: np.array([]))


class SimtelIOEventDataWriter:
    """
    Generate a reduced dataset from given simulation event list and save the output to file.

    Attributes
    ----------
    input_files : list
        List of input file paths to process.
    output_file : str
        Path to the output HDF5 file.
    max_files : int, optional
        Maximum number of files to process.
    """

    def __init__(self, input_files, output_file, max_files=100):
        """Initialize the MCEventExtractor with input files, output file, and max file limit."""
        self._logger = logging.getLogger(__name__)
        self.input_files = input_files
        self.output_file = output_file
        self.max_files = max_files if max_files < len(input_files) else len(input_files)
        self.shower = None
        self.n_use = None
        self.shower_id_offset = 0
        self.event_data = ShowerEventData()
        self.triggered_data = TriggeredEventData()
        self.file_names = []

    def process_files(self):
        """Process the input files and store them in an HDF5 file."""
        if not self.input_files:
            self._logger.warning("No input files provided.")
            return

        self.shower_id_offset = 0

        for i, file in enumerate(self.input_files[: self.max_files], start=1):
            self._logger.info(f"Processing file {i}/{self.max_files}: {file}")
            self._process_file(file)
            if i == 1 or len(self.event_data.simulated_energy) >= 1e7:
                self._write_data(mode="w" if i == 1 else "a")
                self.shower_id_offset += len(self.event_data.simulated_energy)
                self._reset_data()

        self._write_data(mode="a")

    def _process_file(self, file):
        """Process a single file and update data lists."""
        with EventIOFile(file) as f:
            for eventio_object in f:
                if isinstance(eventio_object, MCRunHeader):
                    self._process_mc_run_header(eventio_object)
                elif isinstance(eventio_object, MCShower):
                    self._process_mc_shower(eventio_object)
                elif isinstance(eventio_object, MCEvent):
                    self._process_mc_event(eventio_object)
                elif isinstance(eventio_object, ArrayEvent):
                    self._process_array_event(eventio_object)
            self.file_names.append(str(file))

    def _process_mc_run_header(self, eventio_object):
        """Process MC run header and update data lists."""
        mc_head = eventio_object.parse()
        self.n_use = mc_head["n_use"]  # reuse factor n_use needed to extend the values below
        self._logger.info(f"Shower reuse factor: {self.n_use} (viewcone: {mc_head['viewcone']})")

    def _process_mc_shower(self, eventio_object):
        """
        Process MC shower and update shower event list.

        Duplicated entries 'self.n_use' times to match the number simulated events with
        different core positions.
        """
        self.shower = eventio_object.parse()

        self.event_data.simulated_energy = np.append(
            self.event_data.simulated_energy,
            np.full(self.n_use, self.shower["energy"]),
        )
        self.event_data.shower_azimuth = np.append(
            self.event_data.shower_azimuth,
            np.full(self.n_use, self.shower["azimuth"]),
        )
        self.event_data.shower_altitude = np.append(
            self.event_data.shower_altitude,
            np.full(self.n_use, self.shower["altitude"]),
        )

    def _process_mc_event(self, eventio_object):
        """Process MC event and update shower event list."""
        event = eventio_object.parse()

        self.event_data.shower_id = np.append(self.event_data.shower_id, event["shower_num"])
        self.event_data.x_core = np.append(self.event_data.x_core, event["xcore"])
        self.event_data.y_core = np.append(self.event_data.y_core, event["ycore"])
        self.event_data.area_weight = np.append(self.event_data.area_weight, event["aweight"])

    def _process_array_event(self, eventio_object):
        """Process array event and update triggered event list."""
        tracking_positions = []
        previous_index = -1

        for i, obj in enumerate(eventio_object):
            if isinstance(obj, TriggerInformation):
                self._process_trigger_information(obj)

            if isinstance(obj, TrackingPosition):
                tracking_position = obj.parse()
                tracking_positions.append(
                    {
                        "altitude": tracking_position["altitude_raw"],
                        "azimuth": tracking_position["azimuth_raw"],
                    }
                )

            if i < previous_index:
                self._process_tracking_positions(tracking_positions)
                tracking_positions = []  # Reset for the next shower

            previous_index = i

        if tracking_positions:
            self._process_tracking_positions(tracking_positions)

    def _process_tracking_positions(self, tracking_positions):
        """Process the collected tracking positions and update triggered event list."""
        altitudes = [pos["altitude"] for pos in tracking_positions]
        azimuths = [pos["azimuth"] for pos in tracking_positions]
        if isinstance(altitudes[0], list):
            altitudes, azimuths = altitudes[0], azimuths[0]

        print("FFF", altitudes, azimuths)

        # Check if all tracking positions are the same
        if np.allclose(altitudes, altitudes[0], atol=1e-5) and np.allclose(
            azimuths, azimuths[0], atol=1e-5
        ):
            alt_value, az_value = altitudes[0], azimuths[0]
        else:  # Use the mean telescope tracking positions
            self._logger.info("Telescopes have different tracking positions, applying mean.")
            self._logger.warning("Incorrect calculation of az mean")
            alt_value, az_value = np.mean(altitudes), np.mean(azimuths)

        self.triggered_data.array_altitudes = np.append(
            self.triggered_data.array_altitudes, alt_value
        )
        self.triggered_data.array_azimuths = np.append(self.triggered_data.array_azimuths, az_value)

    def _process_trigger_information(self, trigger_info):
        """Process trigger information and update triggered event list."""
        trigger_info = trigger_info.parse()
        telescopes = trigger_info["telescopes_with_data"]
        if len(telescopes) > 0:
            # add offset to obtained unique shower IDs among all files
            self.triggered_data.triggered_id = np.append(
                self.triggered_data.triggered_id, self.shower["shower"] + self.shower_id_offset
            )
            self.triggered_data.triggered_energy = np.append(
                self.triggered_data.triggered_energy, self.shower["energy"]
            )
            self.triggered_data.trigger_telescope_list_list.append(
                np.array(telescopes, dtype=np.int16)
            )

    def _table_descriptions(self):
        """HDF5 table descriptions for shower data, triggered data, and file names."""
        shower_data_desc = {
            "shower_id": tables.Int32Col(),
            "simulated_energy": tables.Float32Col(),
            "x_core": tables.Float32Col(),
            "y_core": tables.Float32Col(),
            "area_weight": tables.Float32Col(),
            "shower_azimuth": tables.Float32Col(),
            "shower_altitude": tables.Float32Col(),
        }
        triggered_data_desc = {
            "triggered_id": tables.Int32Col(),
            "triggered_energy": tables.Float32Col(),
            "array_altitudes": tables.Float32Col(),
            "array_azimuths": tables.Float32Col(),
            "telescope_list_index": tables.Int32Col(),  # Index into VLArray
        }
        file_names_desc = {
            "file_names": tables.StringCol(256),  # Adjust string length as needed
        }
        return shower_data_desc, triggered_data_desc, file_names_desc

    def _tables(self, hdf5_file, data_group, mode="a"):
        """Create or get HD5 tables."""
        shower_data_desc, triggered_data_desc, file_names_desc = self._table_descriptions()

        if mode == "w" or "/data/reduced_data" not in hdf5_file:
            reduced_table = hdf5_file.create_table(
                data_group,
                "reduced_data",
                shower_data_desc,
                "Reduced Shower Data",
                filters=DEFAULT_FILTERS,
            )
        else:
            reduced_table = hdf5_file.get_node("/data/reduced_data")

        if mode == "w" or "/data/triggered_data" not in hdf5_file:
            triggered_table = hdf5_file.create_table(
                data_group,
                "triggered_data",
                triggered_data_desc,
                "Triggered Data",
                filters=DEFAULT_FILTERS,
            )
        else:
            triggered_table = hdf5_file.get_node("/data/triggered_data")

        if mode == "w" or "/data/file_names" not in hdf5_file:
            file_names_table = hdf5_file.create_table(
                data_group, "file_names", file_names_desc, "File Names", filters=DEFAULT_FILTERS
            )
        else:
            file_names_table = hdf5_file.get_node("/data/file_names")
        return reduced_table, triggered_table, file_names_table

    def _write_event_data(self, reduced_table):
        """Fill event data tables."""
        if len(self.event_data.simulated_energy) == 0:
            return
        row = reduced_table.row
        for i, energy in enumerate(self.event_data.simulated_energy):
            row["shower_id"] = (
                self.event_data.shower_id[i] if i < len(self.event_data.shower_id) else 0
            )
            row["simulated_energy"] = energy
            row["x_core"] = self.event_data.x_core[i] if i < len(self.event_data.x_core) else 0
            row["y_core"] = self.event_data.y_core[i] if i < len(self.event_data.y_core) else 0
            row["area_weight"] = (
                self.event_data.area_weight[i] if i < len(self.event_data.area_weight) else 0
            )
            row["shower_azimuth"] = self.event_data.shower_azimuth[i]
            row["shower_altitude"] = self.event_data.shower_altitude[i]
            row.append()
        reduced_table.flush()

    def _writer_triggered_data(self, triggered_table, vlarray):
        """Fill triggered event data tables."""
        # Get or create VLArray for telescope lists
        if len(self.triggered_data.triggered_id) == 0:
            return
        row = triggered_table.row
        start_idx = vlarray.nrows
        for i, triggered_id in enumerate(self.triggered_data.triggered_id):
            row["triggered_id"] = triggered_id
            row["triggered_energy"] = self.triggered_data.triggered_energy[i]
            row["array_altitudes"] = self.triggered_data.array_altitudes[i]
            row["array_azimuths"] = self.triggered_data.array_azimuths[i]
            row["telescope_list_index"] = start_idx + i  # Index into the VLArray
            row.append()
            vlarray.append(self.triggered_data.trigger_telescope_list_list[i])
        triggered_table.flush()

    def _write_data(self, mode="a"):
        """Write data to HDF5 file."""
        with tables.open_file(self.output_file, mode=mode) as f:
            if mode == "w" or "/data" not in f:
                data_group = f.create_group("/", "data", "Data group")
            else:
                data_group = f.get_node("/data")

            reduced_table, triggered_table, file_names_table = self._tables(f, data_group, mode)

            self._write_event_data(reduced_table)
            if mode == "w" or "/data/trigger_telescope_list_list" not in f:
                vlarray = f.create_vlarray(
                    data_group,
                    "trigger_telescope_list_list",
                    tables.Int16Atom(),
                    "List of triggered telescope IDs",
                )
            else:
                vlarray = f.get_node("/data/trigger_telescope_list_list")
            self._writer_triggered_data(triggered_table, vlarray)

            # Write file names
            if self.file_names:
                row = file_names_table.row
                for name in self.file_names:
                    row["file_names"] = name
                    row.append()
                file_names_table.flush()

    def _reset_data(self):
        """Reset data structures for batch processing."""
        self.event_data = ShowerEventData()
        self.triggered_data = TriggeredEventData()
        self.file_names = []

    def print_dataset_information(self, n_events=10):
        """Print information about the datasets in the generated HDF5 file."""
        try:
            with tables.open_file(self.output_file, mode="r") as reader:
                print("Datasets in file:")
                for node in reader.iter_nodes(reader.root.data):
                    print(f"- {node.name}: shape={getattr(node, 'shape', 'N/A')}")
                    if hasattr(node, "__len__"):
                        print(f"  Length of dataset: {len(node)}")
                        try:
                            if len(node) > 0:
                                print(
                                    f"  First {min(n_events, len(node))} values: "
                                    f"{node[: min(n_events, len(node))]}"
                                )
                        except (TypeError, ValueError, IndexError):  # Not all Items support slicing
                            pass
        except Exception as exc:
            raise ValueError("An error occurred while reading the HDF5 file") from exc

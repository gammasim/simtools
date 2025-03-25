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

from simtools.utils.geometry import calculate_circular_mean

DEFAULT_FILTERS = tables.Filters(complevel=5, complib="zlib", shuffle=True, bitshuffle=False)


@dataclass
class ShowerEventData:
    """Shower event data."""

    simulated_energy: list = field(default_factory=list)
    x_core: list = field(default_factory=list)
    y_core: list = field(default_factory=list)
    shower_azimuth: list = field(default_factory=list)
    shower_altitude: list = field(default_factory=list)
    shower_id: list = field(default_factory=list)
    area_weight: list = field(default_factory=list)

    x_core_shower: list = field(default_factory=list)
    y_core_shower: list = field(default_factory=list)
    core_distance_shower: list = field(default_factory=list)


@dataclass
class TriggeredEventData:
    """Triggered event data."""

    triggered_id: list = field(default_factory=list)
    array_altitudes: list = field(default_factory=list)
    array_azimuths: list = field(default_factory=list)
    trigger_telescope_list_list: list = field(default_factory=list)
    angular_distance: list = field(default_factory=list)


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
        try:
            self.max_files = max_files if max_files < len(input_files) else len(input_files)
        except TypeError as exc:
            raise TypeError("No input files provided.") from exc
        self.shower = None
        self.n_use = None
        self.shower_id_offset = 0
        self.event_data = ShowerEventData()
        self.triggered_data = TriggeredEventData()
        self.file_names = []

    def process_files(self):
        """Process the input files and store them in an HDF5 file."""
        self.shower_id_offset = 0

        for i, file in enumerate(self.input_files[: self.max_files], start=1):
            self._logger.info(f"Processing file {i}/{self.max_files}: {file}")
            self._process_file(file)
            if i == 1 or len(self.event_data.simulated_energy) >= 1e7:
                self._write_data(mode="w" if i == 1 else "a")
                self.shower_id_offset += len(self.event_data.simulated_energy)
                self._reset_data()

        self._write_data(mode="a")

    def get_event_data(self):
        """
        Return shower and triggered event data.

        Returns
        -------
        ShowerEventData, TriggeredEventData
            Shower and triggered event data.
        """
        return self.event_data, self.triggered_data

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

        self.event_data.simulated_energy.extend([self.shower["energy"]] * self.n_use)
        self.event_data.shower_azimuth.extend([self.shower["azimuth"]] * self.n_use)
        self.event_data.shower_altitude.extend([self.shower["altitude"]] * self.n_use)

    def _process_mc_event(self, eventio_object):
        """Process MC event and update shower event list."""
        event = eventio_object.parse()

        self.event_data.shower_id.append(event["shower_num"])
        self.event_data.x_core.append(event["xcore"])
        self.event_data.y_core.append(event["ycore"])
        self.event_data.area_weight.append(event["aweight"])

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
                print("AAAAAAA")
                self._process_tracking_positions(tracking_positions)
                tracking_positions = []  # Reset for the next shower

            previous_index = i

        if tracking_positions:
            self._process_tracking_positions(tracking_positions)

    def _process_tracking_positions(self, tracking_positions):
        """
        Process collected tracking positions and update triggered event list.

        Use mean telescope tracking positions, averaged over all triggered telescopes.
        """
        altitudes = [pos["altitude"] for pos in tracking_positions]
        azimuths = [pos["azimuth"] for pos in tracking_positions]
        # TODO understand when this is the case
        # if isinstance(altitudes[0], list):
        #    altitudes, azimuths = altitudes[0], azimuths[0]

        print("AAAA", altitudes, np.mean(altitudes))
        self.triggered_data.array_altitudes.append(np.mean(altitudes))
        self.triggered_data.array_azimuths.append(calculate_circular_mean(azimuths))

    def _process_trigger_information(self, trigger_info):
        """Process trigger information and update triggered event list."""
        trigger_info = trigger_info.parse()
        telescopes = trigger_info["triggered_telescopes"]
        if len(telescopes) > 0:
            # add offset to obtained unique shower IDs among all files
            self.triggered_data.triggered_id.append(self.shower["shower"] + self.shower_id_offset)
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
            "array_altitudes": tables.Float32Col(),
            "array_azimuths": tables.Float32Col(),
            "telescope_list_index": tables.Int32Col(),  # Index into VLArray
        }
        file_names_desc = {
            "file_names": tables.StringCol(256),  # Adjust string length as needed
        }
        return shower_data_desc, triggered_data_desc, file_names_desc

    def _tables(self, hdf5_file, data_group, mode="a"):
        """Create or get HDF5 tables."""
        descriptions = self._table_descriptions()
        table_names = ["reduced_data", "triggered_data", "file_names"]

        table_dict = {}
        for name, desc in zip(table_names, descriptions):
            path = f"/data/{name}"
            table_dict[name] = (
                hdf5_file.create_table(
                    data_group, name, desc, name.replace("_", " ").title(), filters=DEFAULT_FILTERS
                )
                if mode == "w" or path not in hdf5_file
                else hdf5_file.get_node(path)
            )

        return table_dict["reduced_data"], table_dict["triggered_data"], table_dict["file_names"]

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
            row["shower_azimuth"] = (
                self.event_data.shower_azimuth[i] if i < len(self.event_data.shower_azimuth) else 0
            )
            row["shower_altitude"] = (
                self.event_data.shower_altitude[i]
                if i < len(self.event_data.shower_altitude)
                else 0
            )
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
            row["array_altitudes"] = (
                self.triggered_data.array_altitudes[i]
                if i < len(self.triggered_data.array_altitudes)
                else 0
            )
            row["array_azimuths"] = (
                self.triggered_data.array_azimuths[i]
                if i < len(self.triggered_data.array_azimuths)
                else 0
            )
            row["telescope_list_index"] = start_idx + i  # Index into the VLArray
            row.append()
            vlarray.append(
                self.triggered_data.trigger_telescope_list_list[i]
                if i < len(self.triggered_data.trigger_telescope_list_list)
                else []
            )
        triggered_table.flush()

    def _write_data(self, mode="a"):
        """Write data to HDF5 file."""
        with tables.open_file(self.output_file, mode=mode) as f:
            data_group = (
                f.create_group("/", "data", "Data group")
                if mode == "w" or "/data" not in f
                else f.get_node("/data")
            )

            reduced_table, triggered_table, file_names_table = self._tables(f, data_group, mode)
            self._write_event_data(reduced_table)

            vlarray = (
                f.create_vlarray(
                    data_group,
                    "trigger_telescope_list_list",
                    tables.Int16Atom(),
                    "List of triggered telescope IDs",
                )
                if mode == "w" or "/data/trigger_telescope_list_list" not in f
                else f.get_node("/data/trigger_telescope_list_list")
            )
            self._writer_triggered_data(triggered_table, vlarray)

            if self.file_names:
                file_names_table.append([[name] for name in self.file_names])
                file_names_table.flush()

    def _reset_data(self):
        """Reset data structures for batch processing."""
        self.event_data = ShowerEventData()
        self.triggered_data = TriggeredEventData()
        self.file_names = []

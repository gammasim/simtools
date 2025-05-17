"""Generate a reduced dataset containing mostly shower information and triggered telescopes."""

import json
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
from tables.nodes import filenode

from simtools.corsika.primary_particle import PrimaryParticle
from simtools.simtel.simtel_io_file_info import get_corsika_run_header
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


@dataclass
class SimulationFileInfo:
    """Simulation file information."""

    file_name: list = field(default_factory=list)
    particle_id: list = field(default_factory=list)
    zenith: list = field(default_factory=list)
    azimuth: list = field(default_factory=list)
    nsb_level: list = field(default_factory=list)


class SimtelIOEventDataWriter:
    """Process sim_telarray events and write reduced data to file.

    Extracts essential information from sim_telarray output files:

    - Shower parameters (energy, core location, direction)
    - Trigger patterns
    - Telescope pointing

    Attributes
    ----------
    input_files : list
        List of input file paths to process.
    output_file : str
        Path to the output file.
    max_files : int, optional
        Maximum number of files to process.
    """

    def __init__(self, input_files, output_file, max_files=100):
        """Initialize class."""
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
        self.file_info = SimulationFileInfo()

    def process_files(self, metadata=None):
        """
        Process input files and store them in an file.

        Parameters
        ----------
        metadata : dict, optional
            Metadata to be stored in the output file.

        """
        self.shower_id_offset = 0

        for i, file in enumerate(self.input_files[: self.max_files], start=1):
            self._logger.info(f"Processing file {i}/{self.max_files}: {file}")
            self._process_file(file)
            if i == 1 or len(self.event_data.simulated_energy) >= 1e7:
                self._write_data(mode="w" if i == 1 else "a", metadata=metadata)
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
        self._process_file_info(file)
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

    def _process_mc_run_header(self, eventio_object):
        """Process MC run header and update data lists."""
        mc_head = eventio_object.parse()
        self.n_use = mc_head["n_use"]  # reuse factor n_use needed to extend the values below
        self._logger.info(f"Shower reuse factor: {self.n_use} (viewcone: {mc_head['viewcone']})")

    def _process_file_info(self, file):
        """Process file information and update file info list."""
        run_info = get_corsika_run_header(file)
        self.file_info.file_name.append(str(file))

        # Get particle ID, defaulting to gamma (1) if not found
        particle = PrimaryParticle(
            particle_id_type="eventio_id", particle_id=run_info.get("primary_id", 1)
        )
        self.file_info.particle_id.append(particle.corsika7_id)

        self.file_info.zenith.append(90.0 - np.degrees(run_info["direction"][1]))
        self.file_info.azimuth.append(np.degrees(run_info["direction"][0]))
        self.file_info.nsb_level.append(self._get_preliminary_nsb_level(str(file)))

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

        for _, obj in enumerate(eventio_object):
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

        if tracking_positions:
            self._process_tracking_positions(tracking_positions)

    def _process_tracking_positions(self, tracking_positions):
        """
        Process collected tracking positions and update triggered event list.

        Use mean telescope tracking positions, averaged over all triggered telescopes.
        """
        altitudes = [pos["altitude"] for pos in tracking_positions]
        azimuths = [pos["azimuth"] for pos in tracking_positions]

        self.triggered_data.array_altitudes.append(np.mean(altitudes))
        self.triggered_data.array_azimuths.append(calculate_circular_mean(azimuths))

    def _process_trigger_information(self, trigger_info):
        """Process trigger information and update triggered event list."""
        trigger_info = trigger_info.parse()
        telescopes = trigger_info["triggered_telescopes"]
        if len(telescopes) > 0:
            # add offset to obtain unique shower IDs among all files
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
        file_info_desc = {
            "file_name": tables.StringCol(256),
            "particle_id": tables.Int32Col(),
            "zenith": tables.Float32Col(),
            "azimuth": tables.Float32Col(),
            "nsb_level": tables.Float32Col(),
        }
        return shower_data_desc, triggered_data_desc, file_info_desc

    def _tables(self, output_file, data_group, mode="a"):
        """Create or get HDF5 tables."""
        table_info = zip(
            ["reduced_data", "triggered_data", "file_info"], self._table_descriptions()
        )

        def get_or_create(name, desc):
            path = f"/data/{name}"
            if mode == "w" or path not in output_file:
                return output_file.create_table(
                    data_group, name, desc, name.replace("_", " ").title(), filters=DEFAULT_FILTERS
                )
            return output_file.get_node(path)

        return tuple(get_or_create(name, desc) for name, desc in table_info)

    def _write_file_info(self, file_info_table):
        """Fill file info table."""
        row = file_info_table.row
        for i, file_name in enumerate(self.file_info.file_name):
            row["file_name"] = file_name
            row["particle_id"] = self.file_info.particle_id[i]
            row["zenith"] = self.file_info.zenith[i]
            row["azimuth"] = self.file_info.azimuth[i]
            row["nsb_level"] = self.file_info.nsb_level[i]
            row.append()
        file_info_table.flush()

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

    def _write_metadata(self, file, metadata=None):
        """Write metadata as json-serialized string to HDF5 file.

        Parameters
        ----------
        file : tables.File
            HDF5 file to store metadata attributes
        metadata : dict, optional
            Metadata to be stored in the output file.
        """
        if not metadata:
            return
        node = filenode.new_node(file, where="/", name="metadata")
        node.write(json.dumps(metadata or {}).encode("utf-8"))

    def _write_data(self, mode="a", metadata=None):
        """Write data and metadata to HDF5 file."""
        with tables.open_file(self.output_file, mode=mode) as f:
            data_group = (
                f.create_group("/", "data", "Data group")
                if mode == "w" or "/data" not in f
                else f.get_node("/data")
            )

            if mode == "w":
                self._write_metadata(f, metadata)

            reduced_table, triggered_table, file_info_table = self._tables(f, data_group, mode)
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
            self._write_file_info(file_info_table)

    def _reset_data(self):
        """Reset data structures for batch processing."""
        self.event_data = ShowerEventData()
        self.triggered_data = TriggeredEventData()
        self.file_info = SimulationFileInfo()

    def _get_preliminary_nsb_level(self, file):
        """
        Return preliminary NSB level from file name.

        Hardwired values are used for "dark", "half", and "full" NSB levels
        (actual values are made up for this example).

        Parameters
        ----------
        file : str
            File name to extract NSB level from.

        Returns
        -------
        float
            NSB level extracted from file name.
        """
        nsb_levels = {"dark": 1.0, "half": 2.0, "full": 5.0}

        for key, value in nsb_levels.items():
            try:
                if key in file.lower():
                    self._logger.warning(f"NSB level set to hardwired value of {value}")
                    return value
            except AttributeError as exc:
                raise AttributeError("Invalid file name.") from exc

        self._logger.warning("No NSB level found in file name, defaulting to 1.0")
        return 1.0

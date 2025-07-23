"""Generate a reduced dataset from sim_telarray output files using astropy tables."""

import logging
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.table import Table
from eventio import EventIOFile
from eventio.simtel import (
    ArrayEvent,
    MCEvent,
    MCRunHeader,
    MCShower,
    TrackingPosition,
    TriggerInformation,
)

from simtools.corsika.primary_particle import PrimaryParticle
from simtools.simtel.simtel_io_file_info import get_corsika_run_header
from simtools.simtel.simtel_io_metadata import (
    get_sim_telarray_telescope_id_to_telescope_name_mapping,
)
from simtools.utils.geometry import calculate_circular_mean
from simtools.utils.names import get_common_identifier_from_array_element_name


@dataclass
class TableSchemas:
    """Define schemas for output tables with units."""

    shower_schema = {
        "shower_id": (np.uint32, None),
        "event_id": (np.uint32, None),
        "file_id": (np.uint32, None),
        "simulated_energy": (np.float64, u.TeV),
        "x_core": (np.float64, u.m),
        "y_core": (np.float64, u.m),
        "shower_azimuth": (np.float64, u.deg),
        "shower_altitude": (np.float64, u.deg),
        "area_weight": (np.float64, None),
    }

    trigger_schema = {
        "shower_id": (np.uint32, None),
        "event_id": (np.uint32, None),
        "file_id": (np.uint32, None),
        "array_altitude": (np.float64, u.deg),
        "array_azimuth": (np.float64, u.deg),
        "telescope_list": (str, None),  # Store as comma-separated string
        "telescope_list_common_id": (str, None),  # Store as comma-separated string
    }

    file_info_schema = {
        "file_name": (str, None),
        "file_id": (np.uint32, None),
        "particle_id": (np.uint32, None),
        "energy_min": (np.float64, u.TeV),
        "energy_max": (np.float64, u.TeV),
        "viewcone_min": (np.float64, u.deg),
        "viewcone_max": (np.float64, u.deg),
        "core_scatter_min": (np.float64, u.m),
        "core_scatter_max": (np.float64, u.m),
        "zenith": (np.float64, u.deg),
        "azimuth": (np.float64, u.deg),
        "nsb_level": (np.float64, None),
    }


class SimtelIOEventDataWriter:
    """
    Process sim_telarray events and write tables to file.

    Extracts essential information from sim_telarray output files:

    - Shower parameters (energy, core location, direction)
    - Trigger patterns
    - Telescope pointing

    Attributes
    ----------
    input_files : list
        List of input file paths to process.
    max_files : int, optional
        Maximum number of files to process.
    """

    def __init__(self, input_files, max_files=100):
        """Initialize class."""
        self._logger = logging.getLogger(__name__)
        self.input_files = input_files
        try:
            self.max_files = max_files if max_files < len(input_files) else len(input_files)
        except TypeError as exc:
            raise TypeError("No input files provided.") from exc

        self.n_use = None
        self.shower_data = []
        self.trigger_data = []
        self.file_info = []
        self.telescope_id_to_name = {}

    def process_files(self):
        """
        Process input files and return tables.

        Returns
        -------
        list
            List of astropy tables containing processed data.
        """
        for i, file in enumerate(self.input_files[: self.max_files]):
            self._logger.info(f"Processing file {i + 1}/{self.max_files}: {file}")
            self._process_file(i, file)

        return self.create_tables()

    def create_tables(self):
        """Create astropy tables from collected data."""
        tables = []
        for data, schema, name in [
            (self.shower_data, TableSchemas.shower_schema, "SHOWERS"),
            (self.trigger_data, TableSchemas.trigger_schema, "TRIGGERS"),
            (self.file_info, TableSchemas.file_info_schema, "FILE_INFO"),
        ]:
            table = Table(rows=data, names=schema.keys())
            table.meta["EXTNAME"] = name
            self._add_units_to_table(table, schema)
            tables.append(table)
        return tables

    def _add_units_to_table(self, table, schema):
        """Add units to a single table's columns."""
        for col, (_, unit) in schema.items():
            if unit is not None:
                table[col].unit = unit

    def _process_file(self, file_id, file):
        """Process a single file and update data lists."""
        self._process_file_info(file_id, file)
        with EventIOFile(file) as f:
            for eventio_object in f:
                if isinstance(eventio_object, MCRunHeader):
                    self._process_mc_run_header(eventio_object)
                elif isinstance(eventio_object, MCShower):
                    self._process_mc_shower(eventio_object, file_id)
                elif isinstance(eventio_object, MCEvent):
                    self._process_mc_event(eventio_object)
                elif isinstance(eventio_object, ArrayEvent):
                    self._process_array_event(eventio_object, file_id)

    def _process_mc_run_header(self, eventio_object):
        """Process MC run header and update data lists."""
        mc_head = eventio_object.parse()
        self.n_use = mc_head["n_use"]  # reuse factor n_use needed to extend the values below
        self._logger.info(f"Shower reuse factor: {self.n_use} (viewcone: {mc_head['viewcone']})")

    def _process_file_info(self, file_id, file):
        """Process file information and append to file info list."""
        run_info = get_corsika_run_header(file)
        self.telescope_id_to_name = get_sim_telarray_telescope_id_to_telescope_name_mapping(file)
        particle = PrimaryParticle(
            particle_id_type="eventio_id", particle_id=run_info.get("primary_id", 1)
        )
        self.file_info.append(
            {
                "file_name": str(file),
                "file_id": file_id,
                "particle_id": particle.corsika7_id,
                "energy_min": run_info["E_range"][0],
                "energy_max": run_info["E_range"][1],
                "viewcone_min": run_info["viewcone"][0],
                "viewcone_max": run_info["viewcone"][1],
                "core_scatter_min": run_info["core_range"][0],
                "core_scatter_max": run_info["core_range"][1],
                "zenith": 90.0 - np.degrees(run_info["direction"][1]),
                "azimuth": np.degrees(run_info["direction"][0]),
                "nsb_level": self._get_preliminary_nsb_level(str(file)),
            }
        )

    def _process_mc_shower(self, eventio_object, file_id):
        """
        Process MC shower and update shower event list.

        Duplicated entries 'self.n_use' times to match the number simulated events with
        different core positions.
        """
        shower = eventio_object.parse()

        self.shower_data.extend(
            {
                "shower_id": shower["shower"],
                "event_id": None,  # filled in _process_mc_event
                "file_id": file_id,
                "simulated_energy": shower["energy"],
                "x_core": None,  # filled in _process_mc_event
                "y_core": None,  # filled in _process_mc_event
                "shower_azimuth": np.degrees(shower["azimuth"]),
                "shower_altitude": np.degrees(shower["altitude"]),
                "area_weight": None,  # filled in _process_mc_event
            }
            for _ in range(self.n_use)
        )

    def _process_mc_event(self, eventio_object):
        """
        Process MC event and update shower event list.

        Expected to be called n_use times after _process_shower.
        """
        event = eventio_object.parse()

        shower_data_index = len(self.shower_data) - self.n_use + event["event_id"] % 100

        try:
            if self.shower_data[shower_data_index]["shower_id"] != event["shower_num"]:
                raise IndexError
        except IndexError as exc:
            raise IndexError(
                f"Inconsistent shower and MC event data for shower id {event['shower_num']}"
            ) from exc

        self.shower_data[shower_data_index].update(
            {
                "event_id": event["event_id"],
                "x_core": event["xcore"],
                "y_core": event["ycore"],
                "area_weight": event["aweight"],
            }
        )

    def _process_array_event(self, eventio_object, file_id):
        """Process array event and update triggered event list."""
        tracking_positions = []
        telescopes = []

        for obj in eventio_object:
            if isinstance(obj, TriggerInformation):
                trigger_info = obj.parse()
                telescopes = (
                    trigger_info["triggered_telescopes"]
                    if len(trigger_info["triggered_telescopes"]) > 0
                    else []
                )
            if isinstance(obj, TrackingPosition):
                tracking_position = obj.parse()
                tracking_positions.append(
                    {
                        "altitude": np.degrees(tracking_position["altitude_raw"]),
                        "azimuth": np.degrees(tracking_position["azimuth_raw"]),
                    }
                )

        if len(telescopes) > 0 and tracking_positions:
            self._fill_array_event(
                self._map_telescope_names(telescopes),
                tracking_positions,
                eventio_object.event_id,
                file_id,
            )

    def _fill_array_event(self, telescopes, tracking_positions, event_id, file_id):
        """Add array event triggered events with tracking positions."""
        altitudes = [pos["altitude"] for pos in tracking_positions]
        azimuths = [pos["azimuth"] for pos in tracking_positions]

        self.trigger_data.append(
            {
                "shower_id": self.shower_data[-1]["shower_id"],
                "event_id": event_id,
                "file_id": file_id,
                "array_altitude": float(np.mean(altitudes)),
                "array_azimuth": float(calculate_circular_mean(np.deg2rad(azimuths))),
                "telescope_list": ",".join(map(str, telescopes)),
                "telescope_list_common_id": ",".join(
                    [
                        str(get_common_identifier_from_array_element_name(tel, 0))
                        for tel in telescopes
                    ]
                ),
            }
        )

    def _map_telescope_names(self, telescope_ids):
        """
        Map sim_telarray telescopes IDs to CTAO array element names.

        Parameters
        ----------
        telescope_ids : list
            List of telescope IDs.

        Returns
        -------
        list
            List of telescope names corresponding to the IDs.
        """
        return [
            self.telescope_id_to_name.get(tel_id, f"Unknown_{tel_id}") for tel_id in telescope_ids
        ]

    def _get_preliminary_nsb_level(self, file):
        """
        Return preliminary NSB level from file name.

        Hardwired values are used for "dark", "half", and "full" NSB levels
        (actual values are made up for this example). Will be replaced with
        reading of sim_telarray metadata entry for NSB level (to be implemented,
        see issue #1572).

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

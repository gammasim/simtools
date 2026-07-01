"""Generate a reduced dataset from simulation files (CORSIKA/sim_telarray) using astropy tables."""

import logging
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.table import Table, vstack
from eventio import EventIOFile, iact
from eventio.simtel import (
    ArrayEvent,
    MCEvent,
    MCRunHeader,
    MCShower,
    RunHeader,
    TrackingPosition,
    TriggerInformation,
)

from simtools.corsika.primary_particle import PrimaryParticle
from simtools.sim_events.file_info import get_corsika_run_and_event_headers
from simtools.simtel.simtel_io_metadata import (
    get_sim_telarray_telescope_id_to_telescope_name_mapping,
    read_sim_telarray_metadata,
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

    schemas = {
        "SHOWERS": shower_schema,
        "TRIGGERS": trigger_schema,
        "FILE_INFO": file_info_schema,
    }


class EventDataWriter:
    """
    Process simulation events (CORSIKA/sim_telarray) and write tables to file.

    Extracts essential information from simulation files, including:

    - Shower parameters (energy, core location, direction)
    - Trigger patterns
    - Telescope pointing

    Attributes
    ----------
    input_files : list
        List of input file paths to process.
    max_files : int, optional
        Maximum number of files to process. By default, process all input files.
    """

    def __init__(self, input_files, max_files=None):
        """Initialize class."""
        self._logger = logging.getLogger(__name__)
        self.input_files = input_files
        try:
            number_of_input_files = len(input_files)
        except TypeError as exc:
            raise TypeError("No input files provided.") from exc
        if max_files is not None and max_files < 0:
            raise ValueError("max_files must be non-negative.")
        self.max_files = (
            number_of_input_files if max_files is None else min(max_files, number_of_input_files)
        )

        self.n_use = None
        self.shower_data = []
        self.trigger_data = []
        self.file_info = []
        self.telescope_id_to_name = {}
        self._reset_data()

    def _reset_data(self):
        """Reset accumulated event records and per-file telescope metadata."""
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
            List of tables containing processed data.
        """
        tables_by_name = {name: [] for name in TableSchemas.schemas}
        for tables in self.iter_table_chunks():
            for table in tables:
                tables_by_name[table.meta["EXTNAME"]].append(table)
        return [vstack(tables_by_name[name]) for name in TableSchemas.schemas]

    def iter_table_chunks(self, chunk_size=100_000):
        """Yield bounded table chunks while processing input files."""
        if chunk_size < 1:
            raise ValueError("chunk_size must be greater than zero.")

        yield self.create_tables()  # initialize all output tables, including empty ones
        for file_id, file in enumerate(self.input_files[: self.max_files]):
            self._logger.info(f"Processing file {file_id + 1}/{self.max_files}: {file}")
            self._reset_data()
            run_info = {}
            shower_rows = trigger_rows = 0

            with EventIOFile(file) as eventio_file:
                for eventio_object in eventio_file:
                    if (
                        isinstance(eventio_object, MCShower | iact.EventHeader)
                        and len(self.shower_data) >= chunk_size
                    ):
                        shower_rows += len(self.shower_data)
                        shower_table = self._create_chunk("SHOWERS", self.shower_data)
                        self.shower_data = []
                        yield [shower_table]

                    self._process_eventio_object(eventio_object, file_id, file, run_info)

                    if len(self.trigger_data) >= chunk_size:
                        trigger_rows += len(self.trigger_data)
                        trigger_table = self._create_chunk(
                            "TRIGGERS", self.trigger_data, allow_empty=True
                        )
                        self.trigger_data = []
                        yield [trigger_table]

            self._process_file_info(file_id, file, run_info or None)
            shower_rows += len(self.shower_data)
            trigger_rows += len(self.trigger_data)
            if shower_rows == 0:
                raise ValueError(
                    f"Incomplete reduced event data for '{file}': table 'SHOWERS' contains no rows."
                )
            if trigger_rows == 0:
                self._logger.warning(f"No triggered events found in input file '{file}'.")
            if len(self.file_info) != 1:
                raise ValueError(
                    f"Incomplete reduced event data for '{file}': expected exactly one "
                    f"FILE_INFO row, found {len(self.file_info)}."
                )

            tables = [self._create_chunk("SHOWERS", self.shower_data)]
            if self.trigger_data:
                tables.append(self._create_chunk("TRIGGERS", self.trigger_data, allow_empty=True))
            tables.append(self._create_chunk("FILE_INFO", self.file_info))
            self._reset_data()
            yield tables

    def _create_chunk(self, table_name, data, allow_empty=False):
        """Validate records and create one typed output-table chunk."""
        schema = TableSchemas.schemas[table_name]
        self._validate_records("chunk", table_name, data, schema, allow_empty=allow_empty)
        return self._create_table(data, table_name)

    def create_tables(self):
        """Create tables from collected data."""
        return [
            self._create_table(data, name)
            for name, data in (
                ("SHOWERS", self.shower_data),
                ("TRIGGERS", self.trigger_data),
                ("FILE_INFO", self.file_info),
            )
        ]

    def _create_table(self, data, table_name):
        """Create one typed table and attach its metadata and units."""
        schema = TableSchemas.schemas[table_name]
        table = self._create_typed_table(data, schema, table_name)
        table.meta["EXTNAME"] = table_name
        self._add_units_to_table(table, schema)
        return table

    @staticmethod
    def _create_typed_table(data, schema, table_name):
        """Create a table using the declared schema instead of inferred dtypes."""
        try:
            return Table(
                rows=data,
                names=list(schema),
                dtype=[column_type for column_type, _ in schema.values()],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Failed to create reduced event data table '{table_name}' using its "
                "declared schema."
            ) from exc

    def _add_units_to_table(self, table, schema):
        """Add units to a single table's columns."""
        for col, (_, unit) in schema.items():
            if unit is not None:
                table[col].unit = unit

    def _process_eventio_object(self, eventio_object, file_id, file, run_info):
        """Process one EventIO object and update accumulated records."""
        if isinstance(eventio_object, RunHeader):
            run_info.update(eventio_object.parse())
            self.telescope_id_to_name = get_sim_telarray_telescope_id_to_telescope_name_mapping(
                file
            )
        elif isinstance(eventio_object, MCRunHeader):
            run_info.update(self._process_mc_run_header(eventio_object))
            if not self.telescope_id_to_name:
                self.telescope_id_to_name = get_sim_telarray_telescope_id_to_telescope_name_mapping(
                    file
                )
        elif isinstance(eventio_object, MCShower):
            shower = self._process_mc_shower(eventio_object, file_id)
            if shower.get("primary_id") is not None:
                run_info.setdefault("primary_id", shower["primary_id"])
        elif isinstance(eventio_object, MCEvent):
            self._process_mc_event(eventio_object)
        elif isinstance(eventio_object, ArrayEvent):
            self._process_array_event(eventio_object, file_id)
        elif isinstance(eventio_object, iact.EventHeader):
            self._process_mc_shower_from_iact(eventio_object, file_id)

    @staticmethod
    def _validate_records(file, table_name, records, schema, allow_empty=False):
        """Validate row presence and required values for one output table."""
        if not records and not allow_empty:
            raise ValueError(
                f"Incomplete reduced event data for '{file}': table '{table_name}' "
                "contains no rows."
            )

        incomplete = []
        for row_index, record in enumerate(records):
            missing = [name for name in schema if record.get(name) is None]
            if missing:
                incomplete.append(
                    (row_index, missing, record.get("shower_id"), record.get("event_id"))
                )

        if incomplete:
            fields = sorted({field for _, missing, _, _ in incomplete for field in missing})
            examples = ", ".join(
                f"row={index}, shower_id={shower_id}, event_id={event_id}"
                for index, _, shower_id, event_id in incomplete[:5]
            )
            raise ValueError(
                f"Incomplete reduced event data for '{file}': table '{table_name}' has "
                f"{len(incomplete)} row(s) with unset required field(s) "
                f"{', '.join(fields)}. Examples: {examples}."
            )

    def _process_mc_run_header(self, eventio_object):
        """Process MC run header (sim_telarray file)."""
        mc_head = eventio_object.parse()
        self.n_use = mc_head["n_use"]  # reuse factor n_use needed to extend the values below
        self._logger.info(f"Shower reuse factor: {self.n_use} (viewcone: {mc_head['viewcone']})")
        return mc_head

    def _process_file_info(self, file_id, file, run_info=None):
        """Process file information and append to file info list."""
        if run_info:  # sim_telarray file
            corsika7_id = PrimaryParticle(
                particle_id_type="eventio_id",
                particle_id=run_info.get("primary_id", 1),
            ).corsika7_id
            nsb = self.get_nsb_level_from_sim_telarray_metadata(file)

            e_min, e_max = run_info["E_range"]
            view_cone_min, view_cone_max = run_info["viewcone"]
            core_min, core_max = run_info["core_range"]
            azimuth, el = np.degrees(run_info["direction"])
            zenith = 90.0 - el
        else:  # CORSIKA IACT file
            run_header, event_header = get_corsika_run_and_event_headers(file)
            corsika7_id = int(event_header["particle_id"])
            e_min = event_header["energy_min"]
            e_max = event_header["energy_max"]
            zenith = np.degrees(event_header["zenith"])
            # Rotate to geographic north
            azimuth = np.degrees(
                event_header["azimuth"] - event_header["angle_array_x_magnetic_north"]
            )
            view_cone_min = event_header["viewcone_inner_angle"]
            view_cone_max = event_header["viewcone_outer_angle"]
            core_min = 0.0
            core_max = run_header["x_scatter"] / 1.0e2  # cm to m
            nsb = 0.0

        self.file_info.append(
            {
                "file_name": str(file),
                "file_id": file_id,
                "particle_id": corsika7_id,
                "energy_min": e_min,
                "energy_max": e_max,
                "viewcone_min": view_cone_min,
                "viewcone_max": view_cone_max,
                "core_scatter_min": core_min,
                "core_scatter_max": core_max,
                "zenith": zenith,
                "azimuth": azimuth,
                "nsb_level": nsb,
            }
        )

    def _process_mc_shower(self, eventio_object, file_id):
        """
        Process MC shower from sim_telarray file and update shower event list.

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
        return shower

    def _process_mc_shower_from_iact(self, eventio_object, file_id):
        """
        Process MC shower from IACT file and update shower event list.

        Duplicated entries 'self.n_use' times to match the number simulated events with
        different core positions.
        """
        shower_header = eventio_object.parse()
        self.n_use = int(shower_header["n_reuse"])

        self.shower_data.extend(
            {
                "shower_id": shower_header["event_number"],
                "event_id": shower_header["event_number"] * 100 + i,
                "file_id": file_id,
                "simulated_energy": shower_header["total_energy"],
                "x_core": shower_header["reuse_x"][i] / 1.0e2,
                "y_core": shower_header["reuse_y"][i] / 1.0e2,
                "shower_azimuth": np.degrees(shower_header["azimuth"]),
                "shower_altitude": 90.0 - np.degrees(shower_header["zenith"]),
                "area_weight": 1.0,
            }
            for i in range(self.n_use)
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
        altitudes = []
        azimuths = []
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
                altitudes.append(np.degrees(tracking_position["altitude_raw"]))
                azimuths.append(np.degrees(tracking_position["azimuth_raw"]))

        if len(telescopes) > 0 and altitudes:
            self._fill_array_event(
                self._map_telescope_names(telescopes),
                altitudes,
                azimuths,
                eventio_object.event_id,
                file_id,
            )

    def _fill_array_event(self, telescopes, altitudes, azimuths, event_id, file_id):
        """Add array event triggered events with tracking positions."""
        self.trigger_data.append(
            {
                "shower_id": self.shower_data[-1]["shower_id"],
                "event_id": event_id,
                "file_id": file_id,
                "array_altitude": float(np.mean(altitudes)),
                "array_azimuth": float(np.degrees(calculate_circular_mean(np.deg2rad(azimuths)))),
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

    def get_nsb_level_from_sim_telarray_metadata(self, file):
        """
        Return NSB level from sim_telarray metadata.

        Falls back to preliminary NSB level if not found.

        Parameters
        ----------
        file : Path
            Path to the sim_telarray file.

        Returns
        -------
        float
            NSB level.
        """
        metadata, _ = read_sim_telarray_metadata(file)
        nsb_integrated_flux = metadata.get("nsb_integrated_flux")
        if nsb_integrated_flux is not None:
            try:
                return float(nsb_integrated_flux)
            except (TypeError, ValueError):
                self._logger.warning(
                    f"Invalid nsb_integrated_flux value '{nsb_integrated_flux}' for {file}"
                )

        return self._get_nsb_level_from_file_name(str(file))

    def _get_nsb_level_from_file_name(self, file):
        """
        Return NSB level from file name.

        Hardwired values are used for "dark", "half", "full", and "moon" NSB levels.
        "moon" is treated as equivalent to "half" (0.835).
        Allows to read legacy sim_telarray files without 'nsb_integrated_flux'
        metadata field.

        Parameters
        ----------
        file : str
            File name to extract NSB level from.

        Returns
        -------
        float
            NSB level extracted from file name.

        Raises
        ------
        ValueError
            If no NSB level keyword is found in the file name.
        """
        nsb_levels = {"dark": 0.24, "half": 0.835, "full": 1.2}
        nsb_levels["moon"] = nsb_levels["half"]  # moon uses same level as half

        for key, value in nsb_levels.items():
            try:
                if key in file.lower():
                    self._logger.warning(f"NSB level set to hardwired value of {value} for {file}")
                    return value
            except AttributeError as exc:
                raise AttributeError("Invalid file name.") from exc

        raise ValueError(
            f"Cannot determine NSB level for '{file}': not found in metadata and "
            f"no recognised keyword ('dark', 'half', 'full', 'moon') in file name."
        )

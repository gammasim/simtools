"""Generate a reduced dataset from simulation files (CORSIKA/sim_telarray) using astropy tables."""

import logging
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.table import Table, vstack

from simtools.sim_events.formats.registry import get_reader


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
        "run_number": (np.uint32, None),
        "particle_id": (np.uint32, None),
        "spectral_index": (np.float64, None),
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
    file_format : str
        Registered simulation-file format, by default "eventio".
    """

    def __init__(self, input_files, max_files=None, file_format="eventio"):
        """Initialize class."""
        self._logger = logging.getLogger(__name__)
        self.input_files = input_files
        self.file_format = file_format
        try:
            number_of_input_files = len(input_files)
        except TypeError as exc:
            raise TypeError("No input files provided.") from exc
        if max_files is not None and max_files < 0:
            raise ValueError("max_files must be non-negative.")
        self.max_files = (
            number_of_input_files if max_files is None else min(max_files, number_of_input_files)
        )

        self.shower_data = []
        self.trigger_data = []
        self.file_info = []
        self.input_metadata = []
        self._reset_data()

    def _reset_data(self):
        """Reset accumulated event records and per-file telescope metadata."""
        self.shower_data = []
        self.trigger_data = []
        self.file_info = []

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

        self.input_metadata = []
        self._reset_data()
        yield self.create_tables()  # initialize all output tables, including empty ones
        for file_id, file in enumerate(self.input_files[: self.max_files]):
            self._logger.info(f"Processing file {file_id + 1}/{self.max_files}: {file}")
            self._reset_data()
            reader = get_reader(file, self.file_format)
            counts = dict.fromkeys(TableSchemas.schemas, 0)
            data = {
                "SHOWERS": self.shower_data,
                "TRIGGERS": self.trigger_data,
                "FILE_INFO": self.file_info,
            }
            for table_name, row in reader.iter_records(file_id=file_id):
                data[table_name].append(row)
                counts[table_name] += 1
                if table_name != "FILE_INFO" and len(data[table_name]) >= chunk_size:
                    yield [self._create_chunk(table_name, data[table_name])]
                    data[table_name].clear()
            self._validate_file_chunk_data(file, counts["SHOWERS"], counts["TRIGGERS"])
            self.input_metadata.append(reader.read_metadata())
            yield [self._create_chunk(name, rows) for name, rows in data.items() if rows]
            self._reset_data()

    def _validate_file_chunk_data(self, file_name, shower_rows, trigger_rows):
        """Validate per-file row counts before writing final chunk tables."""
        if shower_rows == 0:
            raise ValueError(
                f"Incomplete reduced event data for '{file_name}': "
                f"table 'SHOWERS' contains no rows."
            )
        if trigger_rows == 0:
            self._logger.warning(f"No triggered events found in input file '{file_name}'.")
        if len(self.file_info) != 1:
            raise ValueError(
                f"Incomplete reduced event data for '{file_name}': expected exactly one "
                f"FILE_INFO row, found {len(self.file_info)}."
            )

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
        if table_name == "FILE_INFO":
            data = [{**row, "run_number": row.get("run_number", 0)} for row in data]
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

    def get_simulation_input_metadata(self):
        """Return rich provenance records for processed input files."""
        return list(self.input_metadata)

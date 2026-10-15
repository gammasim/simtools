# Configuration and simulation-file formats

simtools resolves simulation models and physical run settings before writing a
simulation program's configuration. Configuration writers translate these values
into native files. Result readers translate native output into the quantities
used by simtools. These two selections are independent.

The supplied writers support CORSIKA7, sim_telarray, and the LightEmission
programs `ff-1m` and `xyzls`. The supplied `eventio` reader supports current
CORSIKA IACT and sim_telarray files. Registration allows a new format to be added;
it does not add support for running another simulation program.

## Shared run settings

`SimulationParameters` holds run number, particle identity, pointing in degrees,
shower counts, event counts including shower reuse, viewcone, and calibration
mode. Telescope simulations supplied with a shower file obtain these values
from its reader. They do not need a CORSIKA configuration object.

Model objects retain model parameters and table access. `SimtelModelWriter`
keeps native table conversions and configuration-writing state separately.
Writing additional site or calibration parameters does not change the telescope
model's parameter dictionary.

## Add a configuration writer

1. Put the native configuration syntax and table conversions in a module for
   the simulation software. Accept resolved models rather than fetching model
   parameters again.
2. Implement `write_config_file(additional_models=None, label=None)` for
   individual models, and `export_config_files()` for an array. Return the
   written configuration path from `write_config_file`.
3. Register the writer with `register_model_writer(software_name, writer_class)`.
4. Select it with `model.write_config_file(simulation_software=software_name)`
   or `array_model.export_config_files(simulation_software=software_name)`.
5. Test native syntax, unit conversions, and preservation of model parameters.
   Add an API reference entry and a real-file integration example when the
   format is supported.

For example, after implementing a writer:

```python
from simtools.simulation.configuration import register_model_writer

register_model_writer("example_telescope", ExampleTelescopeWriter)
configuration_path = telescope_model.write_config_file(simulation_software="example_telescope")
```

A shower writer is registered separately with `register_shower_writer`.
Its factory accepts `array_model`, `run_number`, and `label`, and supplies
`simulation_parameters` alongside its native configuration. CORSIKA7 is the
default selected by `get_shower_configuration`.

For calibration light, `register_light_source_writer` accepts a factory taking
the physical `SimulatorLightEmission` setup. Its writer implements
`make_command(iact_output)`. Select it through `light_source_software` in the
light-source configuration. Source position, pointing, wavelength, intensity,
and event numbering remain in the simulation setup; native LightEmission
options and pulse/angular-distribution tables belong to its writer.

## Add a result reader

1. Put native parsing and unit/coordinate conversions in
   `simtools.sim_events.formats`. Use `EventioReader` as the current example.
2. Implement the reader methods needed by the workflow, described below.
3. Register it with `register_reader(format_name, reader_class)`.
4. Select the format using `simulation_file_format` in an application
   configuration, or `file_format` in `EventDataWriter` and
   `Simulator.write_reduced_event_lists`. The default is `eventio`; file
   suffixes do not select a reader.
5. Test small generated inputs with known energies, directions, reuse positions,
   and telescope identifiers. Check the reduced tables and metadata against
   these values. Real simulation-file compatibility belongs in integration tests.

```python
from simtools.sim_events.formats.registry import register_reader
from simtools.sim_events.writer import EventDataWriter

register_reader("example_format", ExampleReader)
tables = EventDataWriter(input_files, file_format="example_format").process_files()
```

Register a format before calling the consuming workflow. Registration applies
to the current Python process. For command-line applications and parallel workers,
add the factory to the supplied reader registry so each process imports it.
Expose the new module in the API reference.
Unknown selections and duplicate registrations raise `ValueError`.

### Reader methods and quantities

- `read_run_number()` returns the run number, or `None` when absent.
- `count_events()` returns shower and event counts, counting shower reuse as
  separate events where available.
- `iter_records(file_id)` yields `(table_name, row_dictionary)` pairs for
  `SHOWERS`, `TRIGGERS`, and `FILE_INFO`. Supply the complete fields defined in
  `TableSchemas`, with scalar values in its units: energy in TeV, distances in
  metres, and angles in degrees. The reduced tables currently use CORSIKA7
  particle identifiers; convert other native identifiers at the reader boundary.
  Preserve shower/event identifiers and use the supplied `file_id`.
- Supply one `FILE_INFO` row per file and all simulated shower reuse positions,
  including events without an array trigger. Trigger rows contain comma-separated
  native telescope IDs and common simtools telescope IDs.
- `read_metadata()` returns the per-file provenance dictionary after reading:
  native run information, source filename and identifiers, and any available
  telescope metadata. Supply `simulation_software` as a software-name string
  when the source differs from sim_telarray. Keep native header dictionaries here, outside the common
  event rows.
- For a shower file supplied to telescope simulation,
  `read_simulation_parameters()` returns the arguments for
  `SimulationParameters`, excluding `run_number`, `run_mode`, and
  `use_curved_atmosphere`. It also supplies `observation_levels` in metres for
  comparison with the site model. Convert pointing to geographic azimuth and
  zenith in degrees.

A new reader does not have to manufacture eventio headers. The existing
`file_info` native-header convenience functions serve current CORSIKA/eventio
callers; new workflows should use the physical settings and event records.

`EventDataWriter` creates typed tables, validates records, and writes bounded
chunks independently of native parsing. Format readers must keep their own
buffers bounded; `EventioReader` retains at most one shower's reuse positions.
The HDF5 product and its units remain the same when an input reader changes.

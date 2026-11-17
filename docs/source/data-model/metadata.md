# Metadata

Metadata in simtools records the context required to interpret and reproduce simulation outputs.
It complements the event or table payload and is required for traceability across production steps.

## Metadata scopes

simtools workflows use metadata at different scopes:

- **Run or file scope**: software versions, model version, site, array layout, and configuration
  context.
- **Table scope**: column definitions, units, axis semantics, and workflow parameters used to
  derive tabular products.
- **Instrument scope**: telescope- or component-specific keys attached to simulation outputs.

## Provenance requirements

Metadata should allow users to answer:

1. Which software stack generated this product?
2. Which simulation model version and parameter set were used?
3. Which production or calibration configuration produced this result?

When these fields are present and consistent, results can be reviewed, compared, and rerun.

## sim_telarray metadata

simtools writes a dedicated metadata block into sim_telarray outputs. The key registry and schema
rules are documented in [sim_telarray metadata](sim_telarray_metadata.md).

## Reduced event-list metadata

Reduced event-list HDF5 files embed two one-record metadata tables:

- `METADATA` is the canonical standard document produced by `MetadataCollector`.
- `SIMULATION_METADATA` contains input run headers, sim_telarray `HistoryMeta`, and the resolved
  site, layout, telescope, calibration, and parameter records. Model-file contents are not
  embedded. Its statistics count shower and trigger records in the reduced output, with
  counts for each input file also recorded in its provenance.

Production job manifests record event counts per output. These are not summed across model
versions: each detector simulation can process the same shower realizations independently.
The CTAO metadata in the sim_telarray input is retained in the reduced file's input provenance
and recorded by output in the job manifest when available. The shared CTAO keyword set follows
the sim_telarray metadata registry.

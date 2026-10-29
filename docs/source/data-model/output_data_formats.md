# Output Data Formats

simtools workflows exchange structured data products between preparation, production, and
validation steps. This page summarizes the main formats and their typical roles.

## Event-level simulation output

- **EventIO / sim_telarray output** (`.simtel`, often compressed): telescope-level simulated events
  and embedded metadata.
- EventIO output is consumed by downstream CTAO tools and by simtools validation utilities.

## Tabular workflow products

- **ECSV** (`.ecsv`): human-readable tables with units and metadata, used for production grids,
  CORSIKA limits, and statistics summaries.
- **HDF5** (`.hdf5`): compact multidimensional data products used for histogram-based workflows and
  intermediate analysis results.

### Trigger histograms

Trigger-histogram products store dense arrays below
`TRIGGER_HISTOGRAM_DENSE/<reference_id>/<name>`.
Each histogram group contains a `values` dataset and one `edges_<axis>` dataset for every axis.
The `axis_names` and `axis_units` attributes are JSON arrays with one entry per axis, or JSON `null`
when metadata is absent; individual units may also be `null`. The parent `TRIGGER_HISTOGRAM_DENSE`
group has a `format_version` attribute set to `2`.

## Configuration and parameter files

- **YAML** (`.yml`, `.yaml`): workflow configuration and schema documents.
- **JSON** (`.json`): model-parameter records and machine-oriented interchange products.

## Selection guidance

- Use ECSV when tables need inspection, version control, or manual review.
- Use HDF5 for large numeric arrays and histogram products.
- Use YAML and JSON for configuration and schema-validated parameter definitions.

For metadata structures attached to these formats, see [Metadata](metadata.md).

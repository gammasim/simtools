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

## Configuration and parameter files

- **YAML** (`.yml`, `.yaml`): workflow configuration and schema documents.
- **JSON** (`.json`): model-parameter records and machine-oriented interchange products.

## Selection guidance

- Use ECSV when tables need inspection, version control, or manual review.
- Use HDF5 for large numeric arrays and histogram products.
- Use YAML and JSON for configuration and schema-validated parameter definitions.

For metadata structures attached to these formats, see [Metadata](metadata.md).

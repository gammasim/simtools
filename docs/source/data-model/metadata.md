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

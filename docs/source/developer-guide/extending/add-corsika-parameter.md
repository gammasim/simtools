# Add a CORSIKA parameter

CORSIKA model parameters are declared in model parameter schemas. A schema describes the
parameter's model value and its CORSIKA input-card name and formatting.

## Implementation

1. Add or update `src/simtools/schemas/model_parameters/<parameter>.schema.yml`.
2. Add a `simulation_software` entry with `name: corsika`, `internal_parameter_name` set to the
   CORSIKA keyword, and a `format` describing serialization. Supported formats are `scalar`,
   `list`, `bool_flag`, `quantity:<unit>`, and `converter:<name>` for a named conversion that
   cannot be represented by a standard format.
3. If using a converter, add it to `CONVERTERS` in `simtools.corsika.input_writer` and cover its
   conversion behavior with a unit test. Keep the selection of physical values in the
   configuration logic; converters only serialize the selected value.
4. Validate the schema with the repository schema checks and add a test that exercises the
   generated CORSIKA card for the parameter.

## Checklist

- The format matches the value shape and unit in the schema.
- The internal keyword matches the CORSIKA manual.
- The generated card preserves the required keyword order and text formatting.
- The parameter can be added by changing schema data alone when no custom conversion is needed.

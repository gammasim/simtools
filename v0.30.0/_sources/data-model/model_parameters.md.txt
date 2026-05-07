# Simulation Model Parameters

## Introduction

Model parameter data structures are defined in schema files.
To ensure consistency and correctness of the model parameters, these schema files are used to validate the model parameter files with `simtools-validate-file-using-schema`.
Strict versioning of schemas and metaschema is introduced to ensure that the model parameters are always compatible with the schema files.

The model parameters are stored in json-style in the [model repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/simulation-models) and database.
A typical model parameter file looks like:

```json
{
    "schema_version": "0.1.0",
    "model_parameter_schema_version": "0.1.0",
    "parameter": "num_gains",
    "instrument": "LSTN-01",
    "site": "North",
    "parameter_version": "1.0.0",
    "unique_id": null,
    "value": 2,
    "unit": null,
    "type": "int64",
    "file": false
}
```

Two types of schema definitions need to be distinguished:

1. **Meta schema:** The meta schema defines the structure of the model parameter file. It is a multi-document YAML file including several schema versions. The meta schema is used to validate the model parameter files. The `schema_version` field is used to identify the correct meta schema.
2. **Model parameter schema:** The model parameter schema defines the properties of the model parameter. It is a human-readable YAML file that describes the model parameter in detail. The `model_parameter_schema_version` field is used to identify the correct model parameter schema (optional field; default value is `0.1.0`).

The metaschema [model_parameter.metaschema.yml](https://github.com/gammasim/simtools/blob/main/src/simtools/schemas/model_parameter.metaschema.yml) defines the JSON data structure used for model parameters. This meta schema defines e.g., that `parameter` must be a string or `file` a boolean. No additional properties are allowed.

Model parameter schemas are defined in YAML format and located in the [src/simtools/schemas/model_parameters](https://github.com/gammasim/simtools/tree/main/src/simtools/schemas/model_parameters) directory. Each schema file specifies fields such as `name`, `type`, `format`, `units`, and applicable telescopes.  Schemas also include metadata on default values, validation rules, data sources, and links to relevant simulation software.  Model parameter descriptions are primarily derived from the [sim_telarray manual](https://www.mpi-hd.mpg.de/hfm/~bernlohr/sim_telarray/), ensuring alignment with its definitions and conventions.

For above example the schema file [num_gains.schema.yml](https://github.com/gammasim/simtools/blob/main/src/simtools/schemas/model_parameters/num_gains.schema.yml) is:

```yaml
%YAML 1.2
---
title: Schema for num_gains model parameter
version: 0.1.0
meta_schema: simpipe-schema
meta_schema_url: https://raw.githubusercontent.com/gammasim/simtools/main/simtools/schemas/model_parameter_and_data_schema.metaschema.yml
meta_schema_version: 0.1.0
name: num_gains
description: Number of different gains the input signal gets digitized.
data:
  - type: int
    unit: dimensionless
    default: 1
    allowed_range:
      min: 1
      max: 2
instrument:
  class: Camera
  type:
    - LSTN
    - LSTS
    - MSTN
    - MSTS
    - SSTS
    - SCTS
activity:
  setting:
    - SetParameterFromExternal
  validation:
    - ValidateParameterByExpert
source:
  - Initial instrument setup
simulation_software:
  - name: sim_telarray
```

Model parameter schema files follow the fixed meta schema (see `meta_schema`, `meta_schema_url`, and `meta_schema_version` fields in above example and [model_parameter_and_data_schema.metaschema.yml](https://github.com/gammasim/simtools/blob/main/src/simtools/schemas/model_parameter_and_data_schema.metaschema.yml)).

## Valid Keys

Valid keys are described in detail in [model_parameter_and_data_schema.metaschema.yml](https://github.com/gammasim/simtools/blob/main/simtools/schemas/model_parameter_and_data_schema.metaschema.yml). The most important parameters are:

### Header section

- `title`: Title of the schema file.
- `version`: Version of this schema file.
- `meta_schema`: Name of the base schema.
- `meta_schema_url`: URL of the base schema.
- `meta_schema_version`: Version of the base schema.

### Parameter description

- `name`: Name of the parameter.
- `description`: Description of the parameter.
- `short_description`: Short description of the parameter (optional).

### Parameter data

The `data` field is used to describe the actual type, format, and allowed values of the parameter.

- `type`: Data type of the parameter.
- `unit`: Units of the parameter (compatible with astropy units).
- `default`: Default value of the parameter.
- `allowed_range`: Allowed range of the parameter.
- `allowed_values`: Allowed values of the parameter. Use if values are not continuous and restricted to a small list of allowed values.

Tables of any dimensions can be described using the `data_table` field. Example:

```yaml
data:
  - type: data_table
    table_columns:
      - name: wavelength
        description: Wavelength.
        required: true
        unit: nm
        type: double
        required_range:
          min: 300.0
          max: 700.0
        input_processing:
          - remove_duplicates
          - sort
      - name: photo_detection_efficiency
        description: Average quantum or photon detection efficiency.
        required: true
        unit: dimensionless
        type: double
        allowed_range:
          min: 0.0
          max: 1.0
```

Input processing in form of sorting, removing of duplicates, etc. can be specified using the `input_processing` field.

:::{Warning}
The `data_table` field is not yet implemented in the simtools schema and parameter files.
:::

### Instrument description

- `class`: Instrument class. Allow values are *Camera*, *Site*, *Structure*, *Telescope*
- `type`: Instrument type following CTAO Naming.
- `site`: CTAO site. Allowed values are *North* and *South*.

## Activity description

Describes setting and validation activities. Each activity corresponds to a workflow as described in the [simtools workflows repository](https://github.com/gammasim/workflows).

:::{Warning}
The implementation of activities and workflows is incomplete and in progress.
:::

## Data source description

Describes the source of the data or parameter (e.g., *Calibration*)

## Simulation software description

Describes the simulation software (e.g., *sim_telarray* or *corsika*) the parameter is used for.

```yaml
simulation_software:
  - name: corsika
```

Allowed is here to add the internal simulation software parameter naming (if different than the name used here in the schema).
This is important to write correct and consistent CORSIKA or sim\_telarray configuration files.

 Example:

```yaml
simulation_software:
  - name: sim_telarray
    internal_parameter_name: secondary_ref_radius
```

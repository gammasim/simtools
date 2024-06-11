# Simulation Model

## Introduction

Simulation model parameters describe properties of all relevant elements of the observatory including site,
telescopes, and calibration devices.
The management, definition, derivation, verification, and validation of the simulation model is central to the functionality of simtools.
Model parameters are centrally stored (in a database and repository) and described and defined by [schema files](https://github.com/gammasim/simtools/tree/main/simtools/schemas/model_parameters).

Examples for model parameters are those describing atmospheric model (profile, transmission) or geomagnetic field characteristics for the site,
descriptions of optical and mechanical properties of the telescope, and detector plan, trigger, and readout settings for the cameras (see [model parameter schema files](https://github.com/gammasim/simtools/tree/main/simtools/schemas/model_parameters) for all model parameters.)

For simplicity, the term parameter is used for values, (multi-dimensional) vectors of values
(e.g., the mirror reflectivity vs wavelength and photon incident angle), functions, and algorithms
(e.g., the telescope trigger algorithm).
Parameters might be fixed in time (e.g., the number of mirrors on a certain telescope), or variable
on different time scales (e.g., compare the degradation of the mirror reflectivity over several months
with the nightly variability of the atmospheric parameters).

The major components to handle the simulation model in simtools are:

1. **Model parameter databases:** The simulation model is stored in [mongoDB databases](databases.md#databases).
The {ref}`db_handler module <DBHANDLER>` provides reading and writing interfaces to the database.
2. **Model parameter management:** The [model parameters module](mcmodel.md#model-parameters) provides interfaces to manage the simulation model parameters.
3. A **model of an array and its elements** consists of the [SiteModel](mcmodel.md#site-model) and several [TelescopeModel](mcmodel.md#telescope-model)`and [CalibrationModel](mcmodel.md#calibration-model) instances.

Simtools includes applications to write new model parameters to the databases and the export or import model parameters from sim_telarray configuration files (see following sections).

Review and revision control of the simulation models uses the [model parameters gitlab repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/model_parameters/-/blob/main/README.md?ref_type=heads). Simtools provides applications to read, write, and update the databases from this repository.

:::{Note}
The simulation model is a central part of the CTAO Simulation Pipeline. The responsibility for the values and the correctness of the simulation model parameters is with the CTAO Simulation Team.
:::

## Description of model parameters

Schema files define all simulation model parameters are part of [simtools](https://github.com/gammasim/simtools) and can be found in [simtools/schemas/model_parameters](https://github.com/gammasim/simtools/tree/main/simtools/schemas/model_parameters).

These files describe the simulation model parameters including (among others fields) name, type, format, units, applicable telescopes, and parameter description.
They include information about setting and validation activities, data sources, and simulation software.
The schema files and especially the model parameter descriptions are derived from (and planned to be synchronized with) the [sim_telarray manual](https://www.mpi-hd.mpg.de/hfm/~bernlohr/sim_telarray/).

The schema files are in human readable yaml format and follow a fixed meta metaschema (see [model_parameter_and_data_schema.metaschema.yml](https://github.com/gammasim/simtools/blob/main/simtools/schemas/model_parameter_and_data_schema.metaschema.yml); found in simtools schema directory).

### Example

The following example is for a single parameter description.

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

### Valid Keys

Valid keys are described in detail in [model_parameter_and_data_schema.metaschema.yml](https://github.com/gammasim/simtools/blob/main/simtools/schemas/model_parameter_and_data_schema.metaschema.yml). The most important parameters are:

#### Header section

- `title`: Title of the schema file.
- `version`: Version of this schema file.
- `meta_schema`: Name of the base schema.
- `meta_schema_url`: URL of the base schema.
- `meta_schema_version`: Version of the base schema.

#### Parameter description

- `name`: Name of the parameter.
- `description`: Description of the parameter.
- `short_description`: Short description of the parameter (optional).

#### Parameter data

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

#### Instrument description

- `class`: Instrument class. Allow values are *Camera*, *Site*, *Structure*, *Telescope*
- `type`: Instrument type following CTAO Naming.
- `site`: CTAO site. Allowed values are *North* and *South*.

### Activity description

Describes setting and validation activities. Each activity corresponds to a workflow as described in the [simtools workflows repository](https://github.com/gammasim/workflows).

:::{Warning}
The implementation of activities and workflows is incomplete and in progress.
:::

### Data source description

Describes the source of the data or parameter (e.g., *Calibration*)

### Simulation software description

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

## Data structure for model parameters

The model parameters are stored in json-style in the [model repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/model_parameters) and database.
A typical model parameter file looks like:

```json
{
    "parameter": "num_gains",
    "instrument": "LSTN-01",
    "site": "North",
    "version": "2024-02-01",
    "value": 2,
    "unit": null,
    "type": "int64",
    "applicable": true,
    "file": false
}
```

## Updating the model database

:::{Danger}
This is for experts only and might impact the simulation model database for all users.
Tests should be done using a sandbox database before applying this to the production databases.
:::

### Update a single model parameter

New model parameter defined in the simtools database format (json file) can be uploaded to the database using the {ref}`simtools-add-value-from-json-to-db <add_value_from_json_to_db>` application.
New data files can be uploaded using {ref}`simtools-add-file-to-db <add_file_to_db>`.

### Upload a complete simulation model from model repository to model database

The application `db_add_model_parameters_from_repository_to_db.py` allows to export the simulation model parameters from a
model repository to the model database for a given model version.
See the [database](databases.md#databases) section for implementation details on databases in simtools.

Example:

```bash
simtools-add_model-parameters-from-repository-to-db \
      --input_path /path/to/repository \
      --db_name new_db_name
```

This application loops over all subdirectories in `input_path` and uploads all json files to the database `new_db_name` (or updates an existing database with the same name):

- subdirectories starting with `OBS` are uploaded to the `sites` collection
- json files from the subdirectory `configuration_sim_telarray` are uploaded to the `configuration_sim_telarray` collection
- `Files` are added to the `files` collection
- all other json files are uploaded to collection defined in the array element description in [simtools/schemas/array_elements.yml](https://github.com/gammasim/simtools/blob/main/simtools/schemas/array_elements.yml). Allowed values are e.g., `telescopes`, `calibration_devices`.

## Import simulation model parameters

### Import Prod6 model parameters

Prod6 model parameters can be extracted from `sim_telarray` using the following commands.

```bash
./sim_telarray/bin/sim_telarray \
   -c sim_telarray/cfg/CTA/CTA-PROD6-LaPalma.cfg \
   -C limits=no-internal -C initlist=no-internal \
   -C list=no-internal -C typelist=no-internal \
   -C maximum_telescopes=30 -DNSB_AUTOSCALE \
   -DNECTARCAM -DHYPER_LAYOUT -DNUM_TELESCOPES=30 \
   /dev/null 2>|/dev/null | grep '(@cfg)'  >| all_telescope_config_la_palma.cfg

./sim_telarray/bin/sim_telarray \
   -c sim_telarray/cfg/CTA/CTA-PROD6-Paranal.cfg \
   -C limits=no-internal -C initlist=no-internal \
   -C list=no-internal -C typelist=no-internal \
   -C maximum_telescopes=87 -DNSB_AUTOSCALE \
   -DNECTARCAM -DHYPER_LAYOUT -DNUM_TELESCOPES=87 \
   /dev/null 2>|/dev/null | grep '(@cfg)'  >| all_telescope_config_paranal.cfg
```

### Import Prod5 model parameters

Prod5 model parameters can be extracted from `sim_telarray` using the following commands.

```bash
./sim_telarray/bin/sim_telarray \
   -c sim_telarray/cfg/CTA/CTA-PROD5-LaPalma.cfg \
   -C limits=no-internal -C initlist=no-internal \
   -C list=no-internal -C typelist=no-internal \
   -C maximum_telescopes=85 -DNSB_AUTOSCALE \
   -DNECTARCAM -DHYPER_LAYOUT -DNUM_TELESCOPES=85 \
   /dev/null 2>|/dev/null | grep '(@cfg)'  >| all_telescope_config_la_palma_prod5.cfg

./sim_telarray/bin/sim_telarray \
   -c sim_telarray/cfg/CTA/CTA-PROD5-Paranal.cfg \
   -C limits=no-internal -C initlist=no-internal \
   -C list=no-internal -C typelist=no-internal \
   -C maximum_telescopes=120 -DNSB_AUTOSCALE \
   -DNECTARCAM -DHYPER_LAYOUT -DNUM_TELESCOPES=120 \
   /dev/null 2>|/dev/null | grep '(@cfg)'  >| all_telescope_config_paranal_prod5.cfg
```

### Import telescope positions

Positions of array elements like telescopes are provided by CTAO in form of tables (typically ecsv files).
To import these positions into the model parameter repository, see the following example:

```bash
simtools-write-array-element-positions-to-repository \
    --input /path/to/positions.txt \
    --repository_path /path/to/repository \
    --model_version 1.0.0 \
    --coordinate_system ground \
    --site North
```

# Simulation model parameters

Simulation model parameters describe all relevant site and telescope properties.
This includes the atmospheric models (profile, transmission) or geomagnetic field properties for the site,
details on optical and mechanical properties of the telescope, plus camera, readout and trigger parameters.

For simplicity, the term parameter is used for values, (multi-dimensional) vectors of values
(e.g., the mirror reflectivity vs wavelength and photon incident angle), functions, and algorithms
(e.g., the telescope trigger algorithm).
Parameters might be fixed in time (e.g., the number of mirrors on a certain telescope), or variable
on different time scales (e.g., compare the degradation of the mirror reflectivity over several months
with the nightly variability of the atmospheric parameters).

The definition, derivation, verification, and validation of the simulation model is central to the functionality of simtools.

## General setup

The most important features of the simulation model are:

- handled by the `model_parameters` module.

## Schema files for a complete description of simulation model parameter

Schema files describing all simulation model parameters are part of [simtools](https://github.com/gammasim/simtools) and can be found in [simtools/schemas/model_parameters](https://github.com/gammasim/simtools/tree/main/simtools/schemas/model_parameters).
These files describe the simulation model parameters including (among others fields) name, type, format, applicable telescopes, and parameter description.
They include information about setting and validation activities, data sources, and simulation software.
The schema files and especially the model parameter descriptions are derived from (and planned to be synchronized with) the [sim_telarray manual](https://www.mpi-hd.mpg.de/hfm/~bernlohr/sim_telarray/).

The files are in human readable yaml format and follow a fixed [json-schema](https://json-schema.org/).
The full description of the schema is in [model_parameter_and_data_schema.metaschema.yml](https://github.com/gammasim/simtools/blob/main/simtools/schemas/model_parameter_and_data_schema.metaschema.yml) (found in simtools schema directory).

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

Valid keys are described in detail in [model_parameter_and_data_schema.metaschema.yml](https://github.com/gammasim/simtools/blob/main/simtools/schemas/model_parameter_and_data_schema.metaschema.yml). The following list gives a short (incomplete) overview of the most important parameters.

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
- `unit`: Units of the parameter (compatible with astropy units). Note that units are explicitly listed in the jsonschema.yml file.
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

#### Instrument description

- `class`: Instrument class. Allow values are *Camera*, *Site*, *Structure*, *Telescope*
- `type`: Instrument type following CTAO Naming.
- `site`: CTAO site. Allowed values are *North* and *South*.

### Activity description

Describes setting and validation activities. Each activity corresponds to a workflow as described in the [simtools workflows repository](https://github.com/gammasim/workflows).

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

## Export simulation model from model repository to model database

TO BE ADDED

## Export simulation model parameters from sim_telarray

All model parameters can be extracted from `sim_telarray` using the following commands.

### Prod6

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

### Prod5

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

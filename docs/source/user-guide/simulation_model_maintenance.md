
# Simulation model maintenance

`simtools` provides functionality to maintain simulation models and derive simulation model parameters
from original sim_telarray configuration files.

```{warning}
This section is for experts only and generally not needed for regular users.
```

## Updating the model database

```{danger}
The following sections describe functionality which might impact the simulation model database for all users.
Generally, model values should not be changed without a merge request in the [simulation models repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/simulation-models.)
Tests should be done using a sandbox database before applying this to the production databases.
```

### Update a single model parameter

New model parameter defined in the simtools database format (json file) can be uploaded to the database using the {ref}`simtools-add-value-from-json-to-db <db_add_value_from_json_to_db>` application.
New data files can be uploaded using {ref}`simtools-add-file-to-db <db_add_file_to_db>`.

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
- json files from the subdirectory `configuration_sim_telarray/configuration_corsika` are uploaded to the `configuration_sim_telarray/configuration_corsika` collection
- `Files` are added to the `files` collection
- all other json files are uploaded to collection defined in the array element description in [simtools/schemas/array_elements.yml](https://github.com/gammasim/simtools/blob/main/simtools/schemas/array_elements.yml). Allowed values are e.g., `telescopes`, `calibration_devices`.

## Import simulation model parameters from sim_telarray

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

## Import telescope positions

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

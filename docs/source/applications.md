(applications)=

# simtools Applications

Applications are python scripts built on the {ref}`Library` that execute a well defined task.
Application scripts can be found in `simtools/applications`.
They are the the building blocks of [Simulation System Workflows](https://github.com/gammasim/workflows).

Important: depending on the installation type, applications are named differently:

- developers (see {ref}`InstallationForDevelopers`) call applications as described throughout this documentation: `python applications/<application name> ....`
- users (see {ref}`InstallationForUsers`) call applications directly as command line tool. Applications names `simtools-<application name` (with all `_` replaced by `-`)

Each application is configured as described in {ref}`Configuration`.
The available arguments can be access by calling the `python applications/<application name> --help`.

Some applications require one or multiple file names as input in the command line. The system will
first search on main simtools directory for these files, and in case it is not found, it will
search into the directories given by the config parameter *model_path*.

Output files of applications are written to `$output_path/$label`, where
*output_path* is a config parameter and *label* is the name of the application. The plots
produced directly by the application are stored in the sub-directory *application-plots*.
High-level data produced intermediately (e.g PSF tables) can be found in the sub-directories relative to
the specific type of application (e.g *ray-tracing* for optics related applications,
*camera-efficiency* for camera efficiency applications etc). All files related to the simulation model (e.g,
sim_telarray config files) are stored in the sub-directory *model*.

Applications found in the *simtools/application/db_development_tools* directory are not intended for
end-users, but for developers working on the database schema.

# List of applications

(add-file-to-db)=

## add_file_to_db

```{eval-rst}
.. automodule:: add_file_to_db
   :members:
```

(add-value-from-json-to-db)=

## add_value_from_json_to_db

```{eval-rst}
.. automodule:: add_value_from_json_to_db
   :members:
```

## calculate_trigger_rate

```{eval-rst}
.. automodule:: calculate_trigger_rate
   :members:
```

## compare_cumulative_psf

```{eval-rst}
.. automodule:: compare_cumulative_psf
   :members:

```

## convert_all_model_parameters_from_simtel

```{eval-rst}
.. automodule:: convert_all_model_parameters_from_simtel
   :members:

```

## convert_model_parameter_from_simtel

```{eval-rst}
.. automodule:: convert_model_parameter_from_simtel
   :members:
```

## derive_mirror_rnda

```{eval-rst}
.. automodule:: derive_mirror_rnda
   :members:
```

## generate_corsika_histograms

```{eval-rst}
.. automodule:: generate_corsika_histograms
   :members:
```

## generate_default_metadata

```{eval-rst}
.. automodule:: generate_default_metadata
   :members:
```

## generate_simtel_array_histograms

```{eval-rst}
.. automodule:: generate_simtel_array_histograms
   :members:
```

## get_file_from_db

```{eval-rst}
.. automodule:: get_file_from_db
   :members:
```

## get_parameter

```{eval-rst}
.. automodule:: get_parameter
   :members:
```

## light_emission

```{eval-rst}
.. automodule:: light_emission
   :members:
```

## make_regular_arrays

```{eval-rst}
.. automodule:: make_regular_arrays
   :members:
```

## plot_array_layout

```{eval-rst}
.. automodule:: plot_array_layout
   :members:
```

## print_array_elements

```{eval-rst}
.. automodule:: print_array_elements
   :members:

```

## produce_array_config

```{eval-rst}
.. automodule:: produce_array_config
   :members:

```

## sim_showers_for_trigger_rates

```{eval-rst}
.. automodule:: sim_showers_for_trigger_rates
   :members:

```

## simulate_prod

```{eval-rst}
.. automodule:: simulate_prod
   :members:
```

## submit_data_from_external

```{eval-rst}
.. automodule:: submit_data_from_external
   :members:

```

## tune_psf

```{eval-rst}
.. automodule:: tune_psf
   :members:

```

## validate_camera_efficiency

```{eval-rst}
.. automodule:: validate_camera_efficiency
   :members:

```

## validate_camera_fov

```{eval-rst}
.. automodule:: validate_camera_fov
   :members:

```

## validate_file_using_schema

```{eval-rst}
.. automodule:: validate_file_using_schema
   :members:

```

## validate_optics

```{eval-rst}
.. automodule:: validate_optics
   :members:
```

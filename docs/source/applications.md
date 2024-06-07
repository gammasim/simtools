(applications)=

# simtools Applications

Applications are python scripts built on the {ref}`Library` that execute a well defined task.
Application scripts can be found in `simtools/applications`.
They are the the building blocks of [Simulation System Workflows](https://github.com/gammasim/workflows).

Important: depending on the installation type, applications are named differently:

- developers (see {ref}`InstallationForDevelopers`) call applications as described throughout this documentation: `python applications/<application name> ....`
- users (see {ref}`InstallationForUsers`) call applications directly as command-line tool. Applications names `simtools-<application name` (with all `_` replaced by `-`)

Each application is configured as described in {ref}`Configuration`.
The available arguments can be access by calling the `python applications/<application name> --help`.

Some applications require one or multiple filenames as input from the command-line options. The system will
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
end users, but for developers working on the database schema.

## List of applications

```{toctree}
:glob: true
:maxdepth: 1

calculate_trigger_rate
compare_cumulative_psf
convert_all_model_parameters_from_simtel
convert_model_parameter_from_simtel
db_add_file_to_db
db_add_value_from_json_to_db
db_get_file_from_db
db_get_parameter_from_db
derive_mirror_rnda
generate_corsika_histograms
generate_default_metadata
generate_simtel_array_histograms
light_emission
generate_regular_arrays
plot_array_layout
print_array_elements
generate_array_config
sim_showers_for_trigger_rates
simulate_prod
submit_data_from_external
tune_psf
validate_camera_efficiency
validate_camera_fov
validate_file_using_schema
validate_optics
```

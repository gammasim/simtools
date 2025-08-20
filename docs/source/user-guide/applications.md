# Applications

`simtools` are applications that execute a well defined task.
The naming convention for simtools is `simtools-<application name>`, where `<application name>` is the name of the application in lower-case snake_case format.

## Configuration

Applications in simtools are configured by the following four equivalent approaches:

1. command-line arguments
2. configuration files (in YAML format)
3. configuration dictionary when calling the {ref}`Configurator <configurationconfigurator>` class
4. environment variables

To illustrate this, the example below sets the path pointing towards the directory for all data products.

Set the output directory using a command-line argument:

```console
<application_name> --output_path <path name>
```

Set the output directory using a configuration file in YAML format:

```yaml
config_file: <path name>
```

Load the YAML configuration file into the application with:

```console
<application_name> --config <my_config.yml>
```

Configuration parameter read from a environmental variable:

```console
EXPORT SIMTOOLS_OUTPUT_PATH="<path name>"
```

Configuration methods can be combined; conflicting configuration settings raise an Exception.
Configuration parameters are generally expected in lower-case snake-make case.
Configuration parameters for each application are printed to screen when executing the application with the `--help` option.
Parameters with the same functionality are named consistently the same among all applications.

## List of applications

```{toctree}
:glob: true
:maxdepth: 1

simtools-calculate-trigger-rate <applications/simtools-calculate-trigger-rate>
simtools-convert-all-model-parameters-from-simtel <applications/simtools-convert-all-model-parameters-from-simtel>
simtools-convert-geo-coordinates-of-array-elements <applications/simtools-convert-geo-coordinates-of-array-elements>
simtools-convert-model-parameter-from-simtel <applications/simtools-convert-model-parameter-from-simtel>
simtools-db-add-file-to-db <applications/simtools-db-add-file-to-db>
simtools-db-add-simulation-model-from-repository-to-db <applications/simtools-db-add-simulation-model-from-repository-to-db>
simtools-db-add-value-from-json-to-db <applications/simtools-db-add-value-from-json-to-db>
simtools-db-get-array-layouts-from-db <applications/simtools-db-get-array-layouts-from-db>
simtools-db-get-file-from-db <applications/simtools-db-get-file-from-db>
simtools-db-get-parameter-from-db <applications/simtools-db-get-parameter-from-db>
simtools-db-inspect-databases <applications/simtools-db-inspect-databases>
simtools-derive-ctao-array-layouts <applications/simtools-derive-ctao-array-layouts>
simtools-derive-mirror-rnda <applications/simtools-derive-mirror-rnda>
simtools-derive-photon-electron-spectrum <applications/simtools-derive-photon-electron-spectrum>
simtools-derive-psf-parameters <applications/simtools-derive-psf-parameters>
simtools-docs-produce-array-element-report <applications/simtools-docs-produce-array-element-report>
simtools-docs-produce-calibration-reports <applications/simtools-docs-produce-calibration-reports>
simtools-docs-produce-model-parameter-reports <applications/simtools-docs-produce-model-parameter-reports>
simtools-docs-produce-simulation-configuration-report <applications/simtools-docs-produce-simulation-configuration-report>
simtools-generate-array-config <applications/simtools-generate-array-config>
simtools-generate-corsika-histograms <applications/simtools-generate-corsika-histograms>
simtools-generate-default-metadata <applications/simtools-generate-default-metadata>
simtools-generate-regular-arrays <applications/simtools-generate-regular-arrays>
simtools-generate-sim-telarray-histograms <applications/simtools-generate-sim-telarray-histograms>
simtools-generate-simtel-event-data <applications/simtools-generate-simtel-event-data>
simtools-maintain-simulation-model-add-production-table <applications/simtools-maintain-simulation-model-add-production-table>
simtools-maintain-simulation-model-compare-productions <applications/simtools-maintain-simulation-model-compare-productions>
simtools-maintain-simulation-model-verify-production-tables <applications/simtools-maintain-simulation-model-verify-production-tables>
simtools-merge-tables <applications/simtools-merge-tables>
simtools-plot-array-layout <applications/simtools-plot-array-layout>
simtools-plot-tabular-data <applications/simtools-plot-tabular-data>
simtools-plot-tabular-data-for-model-parameter <applications/simtools-plot-tabular-data-for-model-parameter>
simtools-print-version <applications/simtools-print-version>
simtools-production-derive-corsika-limits <applications/simtools-production-derive-corsika-limits>
simtools-production-derive-statistics <applications/simtools-production-derive-statistics>
simtools-production-generate-grid <applications/simtools-production-generate-grid>
simtools-production-merge-corsika-limits <applications/simtools-production-merge-corsika-limits>
simtools-run-application <applications/simtools-run-application>
simtools-simulate-calibration-events <applications/simtools-simulate-calibration-events>
simtools-simulate-light-emission <applications/simtools-simulate-light-emission>
simtools-simulate-prod <applications/simtools-simulate-prod>
simtools-simulate-prod-htcondor-generator <applications/simtools-simulate-prod-htcondor-generator>
simtools-submit-array-layouts <applications/simtools-submit-array-layouts>
simtools-submit-data-from-external <applications/simtools-submit-data-from-external>
simtools-submit-model-parameter-from-external <applications/simtools-submit-model-parameter-from-external>
simtools-validate-camera-efficiency <applications/simtools-validate-camera-efficiency>
simtools-validate-camera-fov <applications/simtools-validate-camera-fov>
simtools-validate-cumulative-psf <applications/simtools-validate-cumulative-psf>
simtools-validate-file-using-schema <applications/simtools-validate-file-using-schema>
simtools-validate-optics <applications/simtools-validate-optics>

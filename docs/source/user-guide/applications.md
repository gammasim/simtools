(applications)=

# simtools Applications

Applications are python scripts built on the {ref}`Library` that execute a well defined task.
Application are the building blocks of simtools.

Important: depending on the installation type, applications are named differently:

- users (see {ref}`InstallationForUsers`) call applications directly as command-line tool. Applications names `simtools-<application name` (with all `_` replaced by `-`)
- developers (see [installation for developers](../developer-guide/getting_started.md#devinstallationfordevelopers)) call applications as described throughout this documentation: `python src/simtools/applications/<application name> ....`

Each application is configured as described in {ref}`Configuration`.
The available arguments can be accessed by calling the `<application name> --help`.

Some applications require one or multiple filenames as input from the command-line options. The system will
first search on main simtools directory for these files, and in case it is not found, it will
search into the directories given by the config parameter *model_path*.

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
simtools-db-inspect-databases  <applications/simtools-db-inspect-databases>
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
simtools-generate-simtel-event-data <applications/simtools-generate-simtel-event-data>
simtools-generate-default-metadata <applications/simtools-generate-default-metadata>
simtools-generate-regular-arrays <applications/simtools-generate-regular-arrays>
simtools-generate-sim-telarray-histograms <applications/simtools-generate-sim-telarray-histograms>
simtools-merge-tables <applications/simtools-merge-tables>
simtools-plot-array-layout <applications/simtools-plot-array-layout>
simtools-plot-tabular-data <applications/simtools-plot-tabular-data>
simtools-production-derive-corsika-limits <applications/simtools-production-derive-corsika-limits>
simtools-production-generate-grid <applications/simtools-production-generate-grid>
simtools-production-derive-statistics <applications/simtools-production-derive-statistics>
simtools-print-version <applications/simtools-print-version>
simtools-run-application <applications/simtools-run-application>
simtools-simulate-light-emission <applications/simtools-simulate-light-emission>
simtools-simulate-prod <applications/simtools-simulate-prod>
simtools-simulate-prod-htcondor-generator <applications/simtools-simulate-prod-htcondor-generator>
simtools-submit-array-layouts <applications/simtools-submit-array-layouts>
simtools-submit-data-from-external <applications/simtools-submit-data-from-external>
simtools-submit-model-parameter-from-external <applications/simtools-submit-model-parameter-from-external>
simtools-validate-camera-efficiency <applications/simtools-validate-camera-efficiency>
simtools-validate-camera-fov <applications/simtools-validate-camera-fov>
simtools-validate-cumulative-psf <applications/simtools-validate-cumulative-psf>
simtools-validate-file-using-schema  <applications/simtools-validate-file-using-schema>
simtools-validate-optics <applications/simtools-validate-optics>
simtools-verify-simulation-model-production-tables <applications/simtools-verify-simulation-model-production-tables>

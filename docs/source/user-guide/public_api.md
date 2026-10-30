(public-api)=

# Public API and stability

From version 1.0, simtools follows semantic versioning for the interfaces listed here.
Everything else may change in any release.

## Stable

- **Command-line applications** (`simtools-*`): names, options and exit codes.
  Removed or renamed options are announced in the release notes.
- **Configuration and result files**: the model-parameter, application-workflow and
  data-product schemas (`simtools/schemas`) and the file formats written by the applications
  (production tables, job-grid files, HDF5 histogram files).
- **Simulation model repository layout** as read by `simtools.model_repository`.
- **Extension points**: the interfaces described in the developer guide for simulation
  stages (shower and telescope simulators), ray-tracing backends, event readers, execution
  backends and plot configurations.

## Not stable

- Python modules, classes and functions of the `simtools` package not listed above.
  The API reference documents them for developers; names with a leading underscore are
  private. Import them in your own code at your own risk and pin the simtools version.
- Log messages, file names of temporary files and the layout of output directories
  that are not documented in an application page.

## Changes before 1.0

The release notes list removed and renamed applications, options and file formats.

# Developer Guidelines

This section provides guidelines for developers of simtools. Great that you want to contribute!
Please contact first the simtools team using one of the contact points listed at the
entry page of this documentation before working on simtools.

Simtools follows generally the development guidelines of CTAO and
ctapipe (see [CTAO development documentation](http://cta-computing.gitlab-pages.cta-observatory.org/documentation/developer-documentation/) and [ctapipe development](https://ctapipe.readthedocs.io/en/latest/developer-guide/index.html)).

## Project setup

The main code repository for simtools is on GitHub: [https://github.com/gammasim/simtools](https://github.com/gammasim/simtools)

The main directories of simtools are:

- root directory: [simtools](https://github.com/gammasim/simtools/tree/main/simtools)
- applications (simtools:) [simtools/applications](https://github.com/gammasim/simtools/tree/main/simtools/applications)
- unit and integration tests: [./tests](https://github.com/gammasim/simtools/tree/main/tests)
- documentation: [./docs](https://github.com/gammasim/simtools/tree/main/docs)
- docker files: [./docker](https://github.com/gammasim/simtools/tree/main/docker)

## Python version

The simtools package is currently developed for Python {{ python_min_requires }}.

## Contributing code

It is recommended to discuss any code changes with the simtools team before starting to implement them
(e.g., by opening an issue on GitHub).

The following steps outline how to contribute code to simtools:

01. Set up your coding environment as outlined in the [getting started](getting_started_as_developer.md) section.
02. Start a new feature branch from the main branch (`git checkout -b new-branch-name`).
03. Implement your code changes.
04. Add unit tests for new modules and functions (see [unit test section](testing.md#unit-tests)).
05. Commit your code changes (use meaningful commit messages) and push them to GitHub.
06. Create a draft pull request on GitHub when all features are implemented.
07. Wait for the CI tests to finish and address any issues that arise.
08. After successful tests, mark the pull request as ready for review.
09. Wait for a review of your code and address any issues that arise.
10. After successful review, the pull request can be merged into the main branch.

Note the [guidelines on pull requests](pull_requests.md#pull-requests).

## Writing Applications

Applications are command-line tools that should be built of the simtools library.
Application should not include complex algorithm, this should be done at the module level.

All applications should follow the same structure:

```python
def main():

    # application name
    label = Path(__file__).stem
    # short description of the application
    description = "...."
    # short help on how to use the application
    usage = "....."

    # configuration handling (from command line, config file, etc)
    config = Configurator(label=label, description=description, usage=usage)
    ...
    args_dict, db_dict = config.initialize()

    # generic logger
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    # application code follows
    ...
```

Application handling should be done using the [Configurator](configuration_module.md#configurationconfigurator) class, which allows to set
configurations from the command-line options, configuration files, or environmental variables.
Check the [commandline_parser](configuration_module.md#configurationcommandline_parser) module for generic command-line arguments before introducing new ones in applications

The documentation of application uses the in-line doc string.

Adding an applications requires the following changes:

- add application code to the [simtools/applications](https://github.com/gammasim/simtools/tree/main/simtools/applications)
- add integration tests to [tests/integration_tests](https://github.com/gammasim/simtools/tree/main/tests/integration_tests)
- modify [pyproject.toml file for pip](https://github.com/gammasim/simtools/blob/main/pyproject.toml) (replace "\_" by "-" and add "simtools-" to the application name)
- add application to documentation in [docs/sources/applications.rst](https://github.com/gammasim/simtools/blob/main/docs/source/applications.rst)

## Dependencies

Dependencies on external packages should be kept to a minimum.
Packages are listed twice:

- in the mamba/conda [environment file](https://github.com/gammasim/simtools/blob/main/environment.yml)
- in the [pyproject.toml file for pip](https://github.com/gammasim/simtools/blob/main/pyproject.toml)

Some of the packages installed are used for the development only and not needed for executing
simtools application (see the ordering in sections in pyproject.toml).

## Integration with CORSIKA and sim_telarray

CORSIKA and sim_telarray are external tools to simtools.
Their integration should be
minimally coupled with the rest of the package. The modules that depend directly on these
tools should be connected to the rest of the package through interfaces. This way, it
will be easier to replace these tools in the future.

One example of this approach is
[simulator module](https://github.com/gammasim/simtools/blob/main/simtools/simulator.py),
which connects to the tools used to manage and run simulations.

## Data files

Data files should be kept outside of the simtools repository with the exception of files required for units tests.
These files should be kept at minimum and are stored in the [tests/resources](https://github.com/gammasim/simtools/tree/main/tests/resources) directory.

Data files required by integration tests are downloaded during testing from the simulation model database.

## Input validation

```{warning}
This section is deprecated and will be removed in the future.
```

Any configurable inputs (e.g. physical parameters) to modules
must have them validated. The validation assures that the units, type and
format are correct and also allow for default values.

The configurable input must be passed to classes through a dictionary or a YAML
file. In the case of a dictionary the parameter is generally called config_data, in the
case of a YAML file, config_file.

The function {ref}`gen.collect_data_from_yaml_or_dict <utilsgeneral>`
must be used to read these arguments. It identifies which case was given and
reads it accordingly, returning a dictionary. It also raises an exception in case none are
given and not allow_empty.

The validation of the input is done by the function value_conversion.validate_config_data, which
receives the dictionary with the collected input and a parameter dictionary. The parameter
dictionary is read from a parameter YAML file in the data/parameters directory.
The file is read through the function io.get_data_file("parameters", filename)
(see data files section).

Parameter YAML files contain the list of parameters to be validated and its
properties. See an example below:

```yaml
zenith_angle:
  len: 1
  unit: !astropy.units.Unit {unit: deg}
  default: !astropy.units.Quantity
    value: 20
    unit: !astropy.units.Unit {unit: deg}
  names: ['zenith', 'theta']
```

- len gives the length of the input. If null, any len is accepted.
- unit is the astropy unit
- default must have the same len
- names is a list of acceptable input names. The key in the returned dict will have the name given at the definition of the block (zenith_angle in this example)

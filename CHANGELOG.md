# CHANGELOG

All notable changes to the simtools project will be documented in this file.
Changes for upcoming releases can be found in the [docs/changes](docs/changes) directory.

This changelog is generated using [Towncrier](https://towncrier.readthedocs.io/), see the developer documentation for more information.

<!-- towncrier release notes start -->

## [0.12.0](https://github.com/gammasim/simtools/tree/0.12.0) - 2025-03-11

### Bugfixes

- Bugfix in database-upload related string to list conversion. ([#1423](https://github.com/gammasim/simtools/pull/1423))
- Fix setting of `db_api_authentication_database` through env variable. ([#1425](https://github.com/gammasim/simtools/pull/1425))


## [0.11.0](https://github.com/gammasim/simtools/tree/0.11.0) - 2025-03-05

### Bugfixes

- Fix yamllint options in pre-commit (was essentially switched off). Fix syntax in many YAML files. ([#1404](https://github.com/gammasim/simtools/pull/1404))
- Fix date writing to ensure  RFC 3339-conform date format. ([#1411](https://github.com/gammasim/simtools/pull/1411))
- Fix reading of combined units from model parameter unit string. ([#1415](https://github.com/gammasim/simtools/pull/1415))

### New Features

- Adding two new applications for generating model parameter documentation reports for a given array element either for a given production, or across multiple productions. ([#1185](https://github.com/gammasim/simtools/pull/1185))
- Add building of optimized corsika binaries (e.g., avx2, avx512) and building of images for different corsika versions. ([#1355](https://github.com/gammasim/simtools/pull/1355))
- Add a simtool to derive limits for energy, viewcone and radius for a simulation. ([#1356](https://github.com/gammasim/simtools/pull/1356))
- Add generic `simtools-run-application` to run one or several simtools using a single configuration file. ([#1379](https://github.com/gammasim/simtools/pull/1379))
- Add reading of database collections from model parameter schemas. ([#1380](https://github.com/gammasim/simtools/pull/1380))
- Add module for simtools dependency management. Allows to retrieve e.g. sim_telarray and CORSIKA versions. ([#1383](https://github.com/gammasim/simtools/pull/1383))
- Expanded functionality of application configuration workflows based on the integration test configurations. ([#1389](https://github.com/gammasim/simtools/pull/1389))
- Set user in docker images and add possibility to add user information from env/command line. ([#1400](https://github.com/gammasim/simtools/pull/1400))
- Allow to mark integration tests in their use case config with a corresponding CTAO use case or requirement ID. ([#1405](https://github.com/gammasim/simtools/pull/1405))
- Run check for changelog entry only on PRs ready for review. ([#1416](https://github.com/gammasim/simtools/pull/1416))

### Maintenance

- Maintenance in production configuration removing not needed flags and only use metric files. ([#1376](https://github.com/gammasim/simtools/pull/1376))
- Rename derive-limits application to production-derive-limits. ([#1378](https://github.com/gammasim/simtools/pull/1378))
- Add YAML file with build opts into CORSIKA/sim_telarray images. ([#1382](https://github.com/gammasim/simtools/pull/1382))
- Add unit tests for version.py. ([#1384](https://github.com/gammasim/simtools/pull/1384))
- Improve function naming and unit test coverage in `simtools.utils.names`. ([#1390](https://github.com/gammasim/simtools/pull/1390))
- Change checks for software updates from weekly to monthly. ([#1394](https://github.com/gammasim/simtools/pull/1394))
- Remove duplication in converting sim_telarray string-lists (values, units) to python lists. ([#1395](https://github.com/gammasim/simtools/pull/1395))
- Improve reading of design models; refactoring and improved docstrings for `simtools.utils.names`. ([#1396](https://github.com/gammasim/simtools/pull/1396))

### Simulation model

- Introduce MSTx-NectarCam and MSTx-FlashCam design models for MSTs (allow both sites). ([#1362](https://github.com/gammasim/simtools/pull/1362))


## [0.10.0](https://github.com/gammasim/simtools/tree/0.10.0) - 2025-02-17

### Bugfixes

- Schema validation bugfixes. ([#1330](https://github.com/gammasim/simtools/pull/1330))

### New Features

- Add a simtool to derive (normalize) the single p.e. amplitude spectrum using the `sim_telarray` tool `norm_spe`. ([#1299](https://github.com/gammasim/simtools/pull/1299))
- Add validation of simtools-generated configuration files for sim_telarray with reference files. ([#1322](https://github.com/gammasim/simtools/pull/1322))
- Add schema for fake mirror list. ([#1338](https://github.com/gammasim/simtools/pull/1338))
- Add additional queries to db handler to retrieve model versions etc. ([#1349](https://github.com/gammasim/simtools/pull/1349))
- Add fixed database version for testing (instead of `LATEST`). ([#1365](https://github.com/gammasim/simtools/pull/1365))

### Maintenance

- Major restructering of database routines plus introduction of new simulation model. ([#1316](https://github.com/gammasim/simtools/pull/1316))
- Improve testing of results and allow comparison of lists of floats in json/YAML files. ([#1319](https://github.com/gammasim/simtools/pull/1319))
- Remove database sandboxes . ([#1336](https://github.com/gammasim/simtools/pull/1336))


## [0.9.0](https://github.com/gammasim/simtools/tree/0.9.0) - 2025-01-22

### Bugfixes

- Add correct handling of return codes from simulation software to prevent reports of success even though a job failed (`simulate_prod`) ([#1289](https://github.com/gammasim/simtools/pull/1289))
- Fix setting of activity:end time in the metadata collector. ([#1291](https://github.com/gammasim/simtools/pull/1291))

### New Features

- Add an application to plot tables from a file (from file system or for a model parameter file downloaded from the DB). ([#1267](https://github.com/gammasim/simtools/pull/1267))
- Enhancements to file testing within integration tests. ([#1279](https://github.com/gammasim/simtools/pull/1279))
- Add a tool to prepare job submission with `simulate_prod` for a HTCondor system. ([#1290](https://github.com/gammasim/simtools/pull/1290))
- Remove functionality in `db_handler` to read from simulation model repository. ([#1306](https://github.com/gammasim/simtools/pull/1306))

### Maintenance

- Change dependency from ctapipe to ctapipe-base for ctapipe v0.23 ([#1247](https://github.com/gammasim/simtools/pull/1247))
- Remove pytest dependent from user installation. ([#1286](https://github.com/gammasim/simtools/pull/1286))


## [0.8.2](https://github.com/gammasim/simtools/tree/0.8.2) - 2024-12-03

### Maintenance

- Move simtools package layout to src-layout. ([#1264](https://github.com/gammasim/simtools/pull/1264))


## [v0.8.1](https://github.com/gammasim/simtools/tree/v0.8.1) - 2024-11-26

### Bugfixes

- Fix authentication and writing permits for deployment of simtools to pypi. ([#1256](https://github.com/gammasim/simtools/pull/1256))


## [v0.8.0](https://github.com/gammasim/simtools/tree/v0.8.0) - 2024-11-26

### API Changes

- Improve primary particle API and validate consistency of user input. ([#1127](https://github.com/gammasim/simtools/pull/1127))
- Refactor old style configuration of ray-tracing tools. ([#1142](https://github.com/gammasim/simtools/pull/1142))

### Bugfixes

- Fix triggering of telescopes in light emission package. ([#1128](https://github.com/gammasim/simtools/pull/1128))

### Documentation

- Add missing API documentation for several applications. ([#1194](https://github.com/gammasim/simtools/pull/1194))
- Add Towncrier for CHANGELOG generation for each release. ([#1236](https://github.com/gammasim/simtools/pull/1236))

### New Features

- Added application `submit_model_parameter_from_external.py to submit and validate a new model parameter to SimPipe. ([#1224](https://github.com/gammasim/simtools/pull/1224))
- Add simulation production configuration tool. ([#1244](https://github.com/gammasim/simtools/pull/1244))

### Maintenance

- Update build command for CORSIKA/sim\_telarray to prod6-sc. ([#1235](https://github.com/gammasim/simtools/pull/1235))
- Refactoring of integration test routines into `simtools.testing` modules. ([#1238](https://github.com/gammasim/simtools/pull/1238))

### Simulation model

- Move to strictly using semantic versions for models (remove e.g., 'prod5', 'prod6', 'Latest', 'Released'). ([#1123](https://github.com/gammasim/simtools/pull/1123))
- Add schema for correction of NSB spectrum to the telescope altitude. ([#1171](https://github.com/gammasim/simtools/pull/1171))
- Allow for class Telescope in model parameters. ([#1173](https://github.com/gammasim/simtools/pull/1173))


## [0.7.0](https://github.com/gammasim/simtools/tree/0.7.0) - 2024-11-12

### New Features

- Added application `calculate_trigger_rate` to derive the trigger rate for a given telescope type. ([#801](https://github.com/gammasim/simtools/pull/801))
- Added application `simulate_light_emission` to simulate the light emission from an artificial light source. ([#802](https://github.com/gammasim/simtools/pull/802))

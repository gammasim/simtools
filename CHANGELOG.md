# CHANGELOG

All notable changes to the simtools project will be documented in this file.
Changes for upcoming releases can be found in the [docs/changes](docs/changes) directory.

This changelog is generated using [Towncrier](https://towncrier.readthedocs.io/), see the developer documentation for more information.

<!-- towncrier release notes start -->

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

# CHANGELOG

All notable changes to the simtools project will be documented in this file.
Changes for upcoming releases can be found in the [docs/changes](docs/changes) directory.

This changelog is generated using [Towncrier](https://towncrier.readthedocs.io/), see the developer documentation for more information.

<!-- towncrier release notes start -->

## [v0.19.0](https://github.com/gammasim/simtools/releases/tag/v0.19.0) - 2025-08-27

### Bugfixes

- Fix bug in reading list of array elements from file. Only layout names were used and the list was read not from file but from the model parameters DB. ([#1658](https://github.com/gammasim/simtools/pull/1658))
- Ensure that run numbers start at 1 for cases no offset if given (HT Condor job submission). ([#1686](https://github.com/gammasim/simtools/pull/1686))
- Fix array trigger writing for single telescope simulations. ([#1690](https://github.com/gammasim/simtools/pull/1690))
- Bugfix in shower core calculation in shower coordinates for reduced event data. ([#1703](https://github.com/gammasim/simtools/pull/1703))

### New Features

- Add additional plots for the derivation of CORSIKA limits. ([#1632](https://github.com/gammasim/simtools/pull/1632))
- Add a warning to derived corsika limit if they are close to the limits of the simulations. ([#1643](https://github.com/gammasim/simtools/pull/1643))
- Add an application that merges derived corsika simulation limits for each available grid point. ([#1649](https://github.com/gammasim/simtools/pull/1649))
- Add curvature radius to testeff command. User for calculation of NSB rates per pixel. ([#1651](https://github.com/gammasim/simtools/pull/1651))
- Allow to run applications in container environments using simtools-run-application. Introduce separate module (simtools-runner) and improve support for workflow running. ([#1657](https://github.com/gammasim/simtools/pull/1657))
- Add application to update, modify, and compare production tables.
  Improve documentation for simulation repository maintenance. ([#1660](https://github.com/gammasim/simtools/pull/1660))
- Add application to simulate calibration events (pedestals, dark pedestals, flasher). ([#1663](https://github.com/gammasim/simtools/pull/1663))
- Add functionality to simtools-validate-camera-efficiency to write out model parameter JSON with nsb_pixel_rate. ([#1665](https://github.com/gammasim/simtools/pull/1665))
- Add functionality to compare derived model parameters with DB values to integration tests (examples for simtools-validate-camera-efficiency). ([#1666](https://github.com/gammasim/simtools/pull/1666))
- Add log file inspector for integration tests to fail tests when errors or runtime warnings are reported. ([#1674](https://github.com/gammasim/simtools/pull/1674))
- Add application to simulate flasher with the light emission package. ([#1676](https://github.com/gammasim/simtools/pull/1676))
- Add sim_telarray model files to output of each production run. This asserts reproducibility and simplifies debugging. ([#1689](https://github.com/gammasim/simtools/pull/1689))
- Add MongoDB compound indices to improve query efficiency. ([#1691](https://github.com/gammasim/simtools/pull/1691))
- Add possibility to set random seed for random focal length settings in simtools-derive-mirror-rnda. ([#1702](https://github.com/gammasim/simtools/pull/1702))

### Maintenance

- Apply more consistent key-name convention with lower-case snake_make to application workflow configurations. ([#1636](https://github.com/gammasim/simtools/pull/1636))
- Clarify input data to derivation of CORSIKA limits with `simtools-production-derive-corsika-limit`. ([#1641](https://github.com/gammasim/simtools/pull/1641))
- Change metadata to lower-case snake make. More consistent usage of schema versions. ([#1645](https://github.com/gammasim/simtools/pull/1645))
- Add missing plotting configurations for tabular data. ([#1647](https://github.com/gammasim/simtools/pull/1647))
- Plots are now auto-generated alongside reports. ([#1653](https://github.com/gammasim/simtools/pull/1653))
- Refactoring of ascii input io operations. ([#1667](https://github.com/gammasim/simtools/pull/1667))
- Add a generic writer for structured data for JSON and YAML format. ([#1668](https://github.com/gammasim/simtools/pull/1668))
- Simplification of legend_handlers class. ([#1670](https://github.com/gammasim/simtools/pull/1670))
- Change units of shower- and pointing directions in reduced event data from radians to degrees. ([#1672](https://github.com/gammasim/simtools/pull/1672))
- Generalization of reduced event-data histogram filling (moved from CORSIKA limits derivation modules to generic module). ([#1675](https://github.com/gammasim/simtools/pull/1675))
- Fix hadolint issue and using parameter expansion for replacement in Dockerfile. ([#1680](https://github.com/gammasim/simtools/pull/1680))
- Refactor PSF parameter optimization workflow and improve output. Added a flag to export best parameters as JSON files. ([#1681](https://github.com/gammasim/simtools/pull/1681))
- Revision of main repository readme file and removed duplications. ([#1684](https://github.com/gammasim/simtools/pull/1684))
- Remove DB access of unit test for the `docs_read_parameters` module. ([#1688](https://github.com/gammasim/simtools/pull/1688))
- Maintenance pass through integration test to remove duplicated tests and improve test efficiency. ([#1694](https://github.com/gammasim/simtools/pull/1694))
- Add new sonar configuration ([URL](https://sonar-ctao.zeuthen.desy.de/tutorials?id=gammasim_simtools_0d23837b-8b2d-4e54-9a98-2f1bde681f14)) ([#1695](https://github.com/gammasim/simtools/pull/1695))
- Improve code quality by addressing SonarQube reliability issues. ([#1697](https://github.com/gammasim/simtools/pull/1697))
- Unit tests will now fail in the CI for warnings. ([#1699](https://github.com/gammasim/simtools/pull/1699))
- Fix some security hotspots reported by SonarQube. ([#1700](https://github.com/gammasim/simtools/pull/1700))
- Improve robustness of Linter CI by caching pre-commit installation. ([#1704](https://github.com/gammasim/simtools/pull/1704))

### Simulation model

- Addition of FADC long-sum model parameters. ([#1659](https://github.com/gammasim/simtools/pull/1659))


## [v0.18.0](https://github.com/gammasim/simtools/tree/v0.18.0) - 2025-07-08

### Bugfixes

- Run documentation generation CI on release to obtain current version of docs. ([#1607](https://github.com/gammasim/simtools/pull/1607))
- Fix authentication errors in script to setup a local mongoDB. ([#1610](https://github.com/gammasim/simtools/pull/1610))

### Documentation

- Reorganisation and improvement of the documentation in the GitHub pages. ([#1611](https://github.com/gammasim/simtools/pull/1611))

### New Features

- Use CTAO telescope names when writing out telescope list in reduced event data tables. ([#1616](https://github.com/gammasim/simtools/pull/1616))
- Allow CORSIKA limits to use layouts defined in simulation models database. ([#1619](https://github.com/gammasim/simtools/pull/1619))
- Add application / API to plot tabulated model parameters using default plotting configurations defined in model parameters schemas. ([#1628](https://github.com/gammasim/simtools/pull/1628))
- Allow to use current prod6 simulations for reduced event data generation. ([#1631](https://github.com/gammasim/simtools/pull/1631))
- Add common array elements ID. Add list with triggered common telescope IDs to reduced event data tables. ([#1637](https://github.com/gammasim/simtools/pull/1637))

### Maintenance

- Removed unused ‘critical’ log level. ([#1604](https://github.com/gammasim/simtools/pull/1604))
- Add JSON checker to pre-commit linter. ([#1615](https://github.com/gammasim/simtools/pull/1615))
- Update test files with proton simulations (simtel output) and add provenance information. ([#1617](https://github.com/gammasim/simtools/pull/1617))
- Improve linter CI to print file containing non-ascii characters. ([#1620](https://github.com/gammasim/simtools/pull/1620))
- Replace file lists `tests/resources/simtel_output_files.txt` used for testing by explicitly listing of files. ([#1629](https://github.com/gammasim/simtools/pull/1629))
- Add default plotting configuration for tabular data. Introduce consistent lower-cases for keys in plotting configuration schema. ([#1635](https://github.com/gammasim/simtools/pull/1635))


## [0.17.0](https://github.com/gammasim/simtools/tree/0.17.0) - 2025-06-10

### Bugfixes

- Fix data type of 'dsum_threshold' parameter. ([#1557](https://github.com/gammasim/simtools/pull/1557))
- Add missing neutrons to eventio particle IDs. ([#1584](https://github.com/gammasim/simtools/pull/1584))
- Fix suffix for sim_telarray output files: no correctly `simtel.zst` (was `.zst` only) ([#1588](https://github.com/gammasim/simtools/pull/1588))

### Documentation

- Change documentation theme to pydata and add version switcher. ([#1560](https://github.com/gammasim/simtools/pull/1560))
- Update documentation on how to run jobs using HT Condor. ([#1563](https://github.com/gammasim/simtools/pull/1563))

### New Features

- Add pixel layout plotting. ([#1541](https://github.com/gammasim/simtools/pull/1541))
- Enable docker-image generation for vector-optimized CORSIKA 7.8. ([#1550](https://github.com/gammasim/simtools/pull/1550))
- Add the sequential option to simulations using multipipe. With this option the CORSIKA and sim_telarray instances run in sequential order as far as possible. This mode is useful particularly on the grid, where typically we request a single core per job. ([#1561](https://github.com/gammasim/simtools/pull/1561))
- Add reading and conversion of primary_id from sim_telarray output. ([#1570](https://github.com/gammasim/simtools/pull/1570))
- Update `simtools-generate-simtel-event-data` to use astropy tables. ([#1573](https://github.com/gammasim/simtools/pull/1573))
- Generalize table reading and writing. Add application to merge tables. ([#1580](https://github.com/gammasim/simtools/pull/1580))
- Write reduced event data file as part of the cleanup stage of the simtools-simulate-prod application. ([#1588](https://github.com/gammasim/simtools/pull/1588))
- Use patches for optimized CORSIKA from gitlab repository instead of cloud download. ([#1589](https://github.com/gammasim/simtools/pull/1589))
- New utility simtools-verify-simulation-model-production-tables to verify completeness of model parameter repository. ([#1592](https://github.com/gammasim/simtools/pull/1592))
- Add custom copilot instructions to support AI used by developers. ([#1594](https://github.com/gammasim/simtools/pull/1594))
- Add new application `simtools-submit-array-layouts` to submit new array-layouts definition and validate the telescopes listed. ([#1596](https://github.com/gammasim/simtools/pull/1596))
- Improve layout plotting; adding plotting of background layout. Add plotting of all layouts. ([#1597](https://github.com/gammasim/simtools/pull/1597))
- Improve printout for assertion error for testing. ([#1598](https://github.com/gammasim/simtools/pull/1598))
- Add production version and build options to sim_telarrary metadata. ([#1601](https://github.com/gammasim/simtools/pull/1601))

### Maintenance

- Add unit tests to increase coverage in interpolation handler. ([#1504](https://github.com/gammasim/simtools/pull/1504))
- Docker file improvements: linter application; bugfix in extra flags for optimized CORSIKA compilation. ([#1544](https://github.com/gammasim/simtools/pull/1544))
- Include stdout and stderr in error message of integration test. ([#1555](https://github.com/gammasim/simtools/pull/1555))
- Add pytest-retry package for 'flaky test'. ([#1558](https://github.com/gammasim/simtools/pull/1558))
- Reduce number of log-messages (INFO) for uploaded model parameters to DB. ([#1559](https://github.com/gammasim/simtools/pull/1559))
- Replace the "run_number_start" argument with "run_number_offset" (clearer) and re-introduce the "run_number" argument to allow specifying a run number (to which the offset is added). ([#1562](https://github.com/gammasim/simtools/pull/1562))
- Use telescope position files for setup in illuminator application and set correct obs level. ([#1566](https://github.com/gammasim/simtools/pull/1566))
- Improve the names of the various simulation output files. They now follow the current convention in current productions, they contain a bit more information (e.g., the model version where it wasn't there before), and they are more consistent with each other. ([#1568](https://github.com/gammasim/simtools/pull/1568))
- Improved interface and interpolation routines to derive production configuration. This creates a pipeline for lookup table generation, production grid generation, creating a grid with limits, which is then further complemented by the production statistics per grid point. ([#1569](https://github.com/gammasim/simtools/pull/1569))
- Fix unit test to run with astropy 7.1 (and earlier versions) (see #1574). ([#1575](https://github.com/gammasim/simtools/pull/1575))
- Simplified Sonar setup by moving properties into GitHub workflow. ([#1578](https://github.com/gammasim/simtools/pull/1578))
- Introduce simplified Codecov configuration. ([#1579](https://github.com/gammasim/simtools/pull/1579))
- Copy the CORSIKA log in binary mode instead of text mode. ([#1582](https://github.com/gammasim/simtools/pull/1582))
- Protect against non-unicode characters in CORSIKA log file. ([#1583](https://github.com/gammasim/simtools/pull/1583))
- Remove obsolete code on submission systems in JobManager. ([#1586](https://github.com/gammasim/simtools/pull/1586))
- Change dummy trigger thresholds to easier-to-spot values. ([#1591](https://github.com/gammasim/simtools/pull/1591))
- Fix duplication of some sim_telarray test files in './tests/resources' and corresponding test fixtures. ([#1593](https://github.com/gammasim/simtools/pull/1593))


## [v0.16.0](https://github.com/gammasim/simtools/tree/v0.16.0) - 2025-05-05

### Documentation

- Add a section on mocking to the developer documentation. ([#1516](https://github.com/gammasim/simtools/pull/1516))
- Improve model parameter description (schemas, versions) in documentation. ([#1524](https://github.com/gammasim/simtools/pull/1524))

### New Features

- Add grid generation production application to generate the grid points for Corsika simulations, with energy thresholds, viewcone and radius interpolation. ([#1482](https://github.com/gammasim/simtools/pull/1482))
- Add support for multiple versions and the multipipe functionality of sim_telarray ([#1496](https://github.com/gammasim/simtools/pull/1496))
- Add validation metadata written by sim_telarray with array model parameters. ([#1505](https://github.com/gammasim/simtools/pull/1505))
- Improve robustness of sim_telarray configuration by adding an invalid (dummy) telescope configuration. ([#1509](https://github.com/gammasim/simtools/pull/1509))
- Adding functionality for producing simulation configuration reports. ([#1510](https://github.com/gammasim/simtools/pull/1510))
- Add random seeds from file for instrument configuration. ([#1511](https://github.com/gammasim/simtools/pull/1511))
- Adapt docker files and building workflows to CORSIKA 7.8000. ([#1512](https://github.com/gammasim/simtools/pull/1512))
- Add sim_telarray random seed for instrument configuration to metadata (and metadata tests). ([#1515](https://github.com/gammasim/simtools/pull/1515))
- Integration test with multipipe and support for packing all output files in grid productions. ([#1520](https://github.com/gammasim/simtools/pull/1520))
- Add versioning to model parameter schema files. Implement example for `corsika_starting_grammage`. ([#1523](https://github.com/gammasim/simtools/pull/1523))
- Added an application to produce calibration device reports. ([#1530](https://github.com/gammasim/simtools/pull/1530))
- Allow `corsika_starting_grammage` to be telescope-type dependent. ([#1539](https://github.com/gammasim/simtools/pull/1539))

### Maintenance

- Separate TelescopeModel and SiteModel for improved readability in `sim_telarray` configuration writer. ([#1489](https://github.com/gammasim/simtools/pull/1489))
- Fix licence definition in pyproject.toml to follow PEP 639. ([#1493](https://github.com/gammasim/simtools/pull/1493))
- Improve consistency of spelling of `sim_telarray`. ([#1494](https://github.com/gammasim/simtools/pull/1494))
- Remove test condition in simulator light emission and mock configuration file access. ([#1495](https://github.com/gammasim/simtools/pull/1495))
- Refactoring of event scaling application and moving most of the functionality into a new module. ([#1498](https://github.com/gammasim/simtools/pull/1498))
- Renaming of event scaling methods to production statistics derivation methods. ([#1502](https://github.com/gammasim/simtools/pull/1502))
- Improved naming and docstrings in simulation runners. ([#1519](https://github.com/gammasim/simtools/pull/1519))
- Remove duplication of reading yaml/json files in simtools.utils.general. ([#1522](https://github.com/gammasim/simtools/pull/1522))
- Remove duplication in `make_run_command` in `corsika_simtel_runner` and `simulator_array`. ([#1525](https://github.com/gammasim/simtools/pull/1525))
- Collect pyeventio related functions in new module `simtel_io_file_info`. ([#1531](https://github.com/gammasim/simtools/pull/1531))
- Add new field model_parameter_schema_version to model parameter test files. ([#1535](https://github.com/gammasim/simtools/pull/1535))

### Simulation model

- Change list of applicable array elements and class for `pedestal_events` model parameter. ([#1490](https://github.com/gammasim/simtools/pull/1490))


## [v0.15.0](https://github.com/gammasim/simtools/tree/v0.15.0) - 2025-04-04

### Maintenance

- Skip testing of DB upload for simpipe integration tests. ([#1488](https://github.com/gammasim/simtools/pull/1488))


## [0.14.0](https://github.com/gammasim/simtools/tree/0.14.0) - 2025-04-03

### Bugfixes

- Fix integration tests uploading values / files to sandbox DB. ([#1453](https://github.com/gammasim/simtools/pull/1453))
- Fix building of production images from branches other than `main`. ([#1467](https://github.com/gammasim/simtools/pull/1467))
- Add missing `parameter` definition in model parameter schema. ([#1470](https://github.com/gammasim/simtools/pull/1470))
- Primary degraded map is a parameter applied for all telescope types. ([#1475](https://github.com/gammasim/simtools/pull/1475))
- Fixes a bug in the selection of the triggered events by telescope IDs for the limits calculation. ([#1483](https://github.com/gammasim/simtools/pull/1483))
- Fix how integration tests are marked for use cases (used in SimPipe AIV). ([#1484](https://github.com/gammasim/simtools/pull/1484))

### New Features

- Added functionality to automate report generation for array elements and parameters. ([#1436](https://github.com/gammasim/simtools/pull/1436))
- Add full set of array pointings to reduced mc event data file if available in the mc header. ([#1439](https://github.com/gammasim/simtools/pull/1439))
- Add fitting of afterpulse spectrum to derive single pe application. ([#1446](https://github.com/gammasim/simtools/pull/1446))
- Run scheduled nightly integration and unit tests on production DB. ([#1454](https://github.com/gammasim/simtools/pull/1454))
- Generalize extraction of MC event tree from sim_telarrary files; generic writer and reader. ([#1458](https://github.com/gammasim/simtools/pull/1458))
- Add functionality to generate reports for site-specific parameters. ([#1459](https://github.com/gammasim/simtools/pull/1459))
- Allow to skip integration tests for production DB. ([#1460](https://github.com/gammasim/simtools/pull/1460))
- Add an application to print the versions of simtools, the DB, sim_telarray, and CORSIKA (`simtools-print-version`). ([#1461](https://github.com/gammasim/simtools/pull/1461))
- Use files to pass filepaths for simtel event data files and telescope IDs in corsika limits derivation tool. ([#1464](https://github.com/gammasim/simtools/pull/1464))
- Retrieve CTAO layout definitions. Merge with model parameter value and write to disk. ([#1465](https://github.com/gammasim/simtools/pull/1465))
- Add tests to ensure that model table files uploaded to DB follow Unicode. ([#1473](https://github.com/gammasim/simtools/pull/1473))
- Add writing of metadata into sim_telarray configuration files. ([#1474](https://github.com/gammasim/simtools/pull/1474))
- Update model parameter definitions to reflect those from most recent sim_telarray release. ([#1476](https://github.com/gammasim/simtools/pull/1476))

### Maintenance

- Remove docker prod image generation; documentation update and improved workflow naming. ([#1447](https://github.com/gammasim/simtools/pull/1447))
- Change requirement of arguments in event scaling application and improve documentation. ([#1449](https://github.com/gammasim/simtools/pull/1449))
- Refactor light emission application and move logic to helper functions. ([#1450](https://github.com/gammasim/simtools/pull/1450))
- Use consistently numpy dtypes for model parameters. ([#1451](https://github.com/gammasim/simtools/pull/1451))
- Remove h5py dependency. ([#1456](https://github.com/gammasim/simtools/pull/1456))
- Store arrays in model parameters as arrays and not as strings. ([#1466](https://github.com/gammasim/simtools/pull/1466))


## [0.13.0](https://github.com/gammasim/simtools/tree/0.13.0) - 2025-03-19

### Bugfixes

- Fix `null` values in YAML-style model parameter schema files. ([#1437](https://github.com/gammasim/simtools/pull/1437))

### New Features

- Improvements for single pe setting workflow including plotting and metadata handling. ([#1398](https://github.com/gammasim/simtools/pull/1398))
- Add an application that allows to extract array and shower data and to save them in hdf5 format for the later calculation of production configuration limits. ([#1402](https://github.com/gammasim/simtools/pull/1402))
- Add integration tests to production image generation. ([#1420](https://github.com/gammasim/simtools/pull/1420))
- Improve reading of single parameter and files from database. Add plotting of tabular data using `parameter_version`. ([#1426](https://github.com/gammasim/simtools/pull/1426))
- Improved plotting of tabular data; adding error bars and schema validation for plotting configuration. ([#1429](https://github.com/gammasim/simtools/pull/1429))
- Add function to get all model parameter data for all model versions. ([#1432](https://github.com/gammasim/simtools/pull/1432))

### Maintenance

- Run unit and integration test on database setup in local test environment. ([#1424](https://github.com/gammasim/simtools/pull/1424))
- Remove tests from pypi package. Remove obsolete `__init__.py` files. ([#1430](https://github.com/gammasim/simtools/pull/1430))
- Change base python version from 3.11 to 3.12. ([#1433](https://github.com/gammasim/simtools/pull/1433))
- Update exporting of sim_telarray model to simtools taking into account the updated file naming (included parameter versions). ([#1438](https://github.com/gammasim/simtools/pull/1438))
- Robust getting of username in metadata collector in case no user is defined on system level. ([#1442](https://github.com/gammasim/simtools/pull/1442))


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

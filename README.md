# simtools

[![LICENSE](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/gammasim/simtools/blob/main/LICENSE)
[![release](https://img.shields.io/github/v/release/gammasim/simtools)](https://github.com/gammasim/simtools/releases)
[![DOI](https://zenodo.org/badge/195011575.svg)](https://zenodo.org/badge/latestdoi/195011575)
[![pypyi](https://badge.fury.io/py/gammasimtools.svg)](https://badge.fury.io/py/gammasimtools)
[![conda](https://anaconda.org/conda-forge/gammasimtools/badges/version.svg)](https://anaconda.org/conda-forge/gammasimtools)

**simtools** is a modular toolkit for managing simulation model parameters, configuring, running, and validating simulation productions for arrays of imaging atmospheric Cherenkov telescopes.

**Documentation**: [simtools documentation](https://gammasim.github.io/simtools/)

**simtools** is part of the [SimPipe](http://cta-computing.gitlab-pages.cta-observatory.org/dpps/simpipe/simpipe/latest/) pipeline of the [Cherenkov Telescope Array Observatory (CTAO)](https://www.cta-observatory.org/).

**Authors**: see the [CITATION.cff](https://github.com/gammasim/simtools/blob/main/CITATION.cff) file.

> **Note**
> simtools is under active development.
> Contact the developers before using it: [simtools-developer@desy.de](mailto:simtools-developer@desy.de)

## Features

- Simulation model libraries and management
- Database interfaces for simulation model parameters (see [CTAO Simulation models](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/simulation-models))
- Tools for preparing, configuring, running, and validating simulation productions
- Applications for deriving and validating model parameters (see CTAO [Model setting and validation workflows](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/simulation-model-parameter-setting))
- Standardized interfaces and outputs independent of the simulation software (e.g., [CORSIKA](https://www.iap.kit.edu/corsika/) air shower simulations and [sim_telarray](https://www.mpi-hd.mpg.de/hfm/~bernlohr/sim_telarray/) telescope simulations)
- I/O and reporting tools for the simulation model and production database

## Citing this Software

Please cite this software if it is used for a publication, see the [Zenodo record](https://doi.org/10.5281/zenodo.6346696) and [CITATION.cff](https://github.com/gammasim/simtools/blob/main/CITATION.cff).

## Software Quality

[![CI](https://github.com/gammasim/simtools/actions/workflows/CI-unittests.yml/badge.svg)](https://github.com/gammasim/simtools/actions/workflows/CI-unittests.yml)
[![CI-integrationtest](https://github.com/gammasim/simtools/actions/workflows/CI-integrationtests.yml/badge.svg)](https://github.com/gammasim/simtools/actions/workflows/CI-integrationtests.yml)
[![CI-docs](https://github.com/gammasim/simtools/actions/workflows/CI-docs.yml/badge.svg)](https://github.com/gammasim/simtools/actions/workflows/CI-docs.yml)

[![Coverage](https://sonar-ctao.zeuthen.desy.de/api/project_badges/measure?project=gammasim_simtools_0d23837b-8b2d-4e54-9a98-2f1bde681f14&metric=coverage&token=sqb_5d1fde56fa060247eee7d5e53fa5ac0a4aabe483)](https://sonar-ctao.zeuthen.desy.de/dashboard?id=gammasim_simtools_0d23837b-8b2d-4e54-9a98-2f1bde681f14)
[![Quality Gate Status](https://sonar-ctao.zeuthen.desy.de/api/project_badges/measure?project=gammasim_simtools_0d23837b-8b2d-4e54-9a98-2f1bde681f14&metric=alert_status&token=sqb_5d1fde56fa060247eee7d5e53fa5ac0a4aabe483)](https://sonar-ctao.zeuthen.desy.de/dashboard?id=gammasim_simtools_0d23837b-8b2d-4e54-9a98-2f1bde681f14)

## Generative AI disclosure

Generative AI tools (including Claude, ChatGPT, and Gemini) were used to assist with code development, debugging, and documentation drafting. All AI-assisted outputs were reviewed, validated, and, where necessary, modified by the authors to ensure accuracy and reliability.

## Acknowledgements

This project is supported by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - project number 460248186 (*PUNCH4NFDI [https://www.punch4nfdi.de/](https://www.punch4nfdi.de/)*).

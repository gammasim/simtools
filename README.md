# simtools

[![LICENSE](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/gammasim/simtools/blob/main/LICENSE)
[![release](https://img.shields.io/github/v/release/gammasim/simtools)](https://github.com/gammasim/simtools/releases)
[![DOI](https://zenodo.org/badge/195011575.svg)](https://zenodo.org/badge/latestdoi/195011575)
[![pypyi](https://badge.fury.io/py/gammasimtools.svg)](https://badge.fury.io/py/gammasimtools)
[![conda](https://anaconda.org/conda-forge/gammasimtools/badges/version.svg)](https://anaconda.org/conda-forge/gammasimtools)

**simtools** is a modular toolkit for managing simulation model parameters, configuring, running, and validating simulation productions for arrays of imaging atmospheric Cherenkov telescopes.

**Documentation**: [simtools documentation](https://gammasim.github.io/simtools/)

**simtools** is part of the [SimPipe](http://cta-computing.gitlab-pages.cta-observatory.org/dpps/simpipe/simpipe/latest/) pipeline of the [Cherenkov Telescope Array Observatory (CTAO)](https://www.cta-observatory.org/).

> **Note**
> simtools is under active development.
> Best to contact the developers before using it: [simtools-developer@desy.de](mailto:simtools-developer@desy.de)

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

[![Quality](https://app.codacy.com/project/badge/Grade/a3f19df7454844059341edd0769e02a7)](https://app.codacy.com/gh/gammasim/simtools/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Coverage](https://codecov.io/gh/gammasim/simtools/graph/badge.svg?token=AYAIRPARCH)](https://codecov.io/gh/gammasim/simtools)
[![Coverage](https://sonar-cta-dpps.zeuthen.desy.de/api/project_badges/measure?project=gammasim_simtools_AY_ssha9WiFxsX-2oy_w&metric=coverage&token=sqb_ef68b705c0bc7f9800f4ff3f087f1d7f7471ff0e)](https://sonar-cta-dpps.zeuthen.desy.de/dashboard?id=gammasim_simtools_AY_ssha9WiFxsX-2oy_w)
[![Quality Gate Status](https://sonar-cta-dpps.zeuthen.desy.de/api/project_badges/measure?project=gammasim_simtools_AY_ssha9WiFxsX-2oy_w&metric=alert_status&token=sqb_ef68b705c0bc7f9800f4ff3f087f1d7f7471ff0e)](https://sonar-cta-dpps.zeuthen.desy.de/dashboard?id=gammasim_simtools_AY_ssha9WiFxsX-2oy_w)

## Acknowledgements

This project is supported by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - project number 460248186 (*PUNCH4NFDI [https://www.punch4nfdi.de/](https://www.punch4nfdi.de/)*).

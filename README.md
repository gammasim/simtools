# simtools

[![LICENSE](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/gammasim/simtools/blob/main/LICENSE)
[![release](https://img.shields.io/github/v/release/gammasim/simtools)](https://github.com/gammasim/simtools/releases)
[![DOI](https://zenodo.org/badge/195011575.svg)](https://zenodo.org/badge/latestdoi/195011575)
[![pypyi](https://badge.fury.io/py/gammasimtools.svg)](https://badge.fury.io/py/gammasimtools)
[![CI](https://github.com/gammasim/simtools/actions/workflows/CI-unittests.yml/badge.svg)](https://github.com/gammasim/simtools/actions/workflows/CI-unittests.yml)
[![CI-integrationtest](https://github.com/gammasim/simtools/actions/workflows/CI-integrationtests.yml/badge.svg)](https://github.com/gammasim/simtools/actions/workflows/CI-integrationtests.yml)
[![CI-docs](https://github.com/gammasim/simtools/actions/workflows/CI-docs.yml/badge.svg)](https://github.com/gammasim/simtools/actions/workflows/CI-docs.yml)
[![Quality](https://app.codacy.com/project/badge/Grade/a3f19df7454844059341edd0769e02a7)](https://app.codacy.com/gh/gammasim/simtools/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Coverage](https://codecov.io/gh/gammasim/simtools/graph/badge.svg?token=AYAIRPARCH)](https://codecov.io/gh/gammasim/simtools)

Simulation tools and applications for the [Cherenkov Telescope Array (CTAO)](https://www.cta-observatory.org).

License: [BSD-3](https://github.com/gammasim/simtools/blob/main/LICENSE)

Useful links:
[Documentation](https://gammasim.github.io/simtools/)
[Source code](https://github.com/gammasim/simtools)
[Issue tracker](https://github.com/gammasim/simtools/issues)
[Citation](https://github.com/gammasim/simtools/blob/main/CITATION.cff)

simtools provides:

- libraries for simulation model definition and management
- model database interface
- tools for the preparation and configuration of simulation productions
- applications for simulation model parameter derivation and validation
- standardized interfaces and data products independent of the underlying simulation software (e.g., [CORSIKA](https://www.iap.kit.edu/corsika/), [sim_telarray](https://www.mpi-hd.mpg.de/hfm/~bernlohr/sim_telarray/))
- I/O and reporting tools for the MC simulation model parameter and production database

simtools is one part of the CTAO Simulation Pipeline, which consist of the following components:

- [CORSIKA](https://www.iap.kit.edu/corsika/) air shower simulation code and the [sim_telarray](https://www.mpi-hd.mpg.de/hfm/~bernlohr/sim_telarray/) telescope simulation code
- [workflows](https://github.com/gammasim/workflows) for setting, derivation and validation of simulation model parameters
- [simulation model parameter](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/model_parameters)
- [databases](https://gammasim.github.io/simtools/databases.html), especially the model parameter database

:::{note}
simtools is currently under heavy development with continuous changes and additions planned.
Please contact the developers before using it: [simtools-developer@desy.de](mailto:simtools-developer@desy.de).
:::

## Quick start

:::{caution}
The pip-installation of simtools provides limited functionality only
and is not as well tested as the conda/mamba installation.
:::

Install simtools with pypi (recommended for users):

```bash
pip install gammasimtools
```

Install simtools with mamba (recommended for developers):

```bash
git clone https://github.com/gammasim/simtools.git
cd simtools
mamba env create -f environment.yml
conda activate simtools
pip install -e .
```

For more detail see the [Getting_Started](https://gammasim.github.io/simtools/getting_started.html) section of the manual.

## Authors

- Raul R Prado (DESY)
- Orel Gueta (DESY)
- Tobias Kleiner (DESY)
- Victor B. Martins (DESY)
- Gernot Maier (DESY)

## Citing this Software

Please cite this software if it use used for a publication, see the [Zenodo record](https://doi.org/10.5281/zenodo.6346696) and [CITATION.cff](https://github.com/gammasim/simtools/blob/main/CITATION.cff) .

## Acknowledgements

This project is supported by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - project number 460248186 (*PUNCH4NFDI [https://www.punch4nfdi.de/](https://www.punch4nfdi.de/)*).

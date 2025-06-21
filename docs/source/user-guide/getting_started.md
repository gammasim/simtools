# Getting Started

The usage of simtools requires the installation and/or access to all [major components](../components/index.md) of simtools.
This includes the [simtools package](#installation) itself, the simulation software
[CORSIKA and sim_telarray](#installation-of-corsika-and-sim_telarray),
and the setting of environment variables to [connect to the simulation models model database](model-database-access).

For developers, please see the [Getting started as developer](../developer-guide/getting_started_as_developer.md) section.

## Installation

These are the options to install simtools:

- [using a docker image](container-docker) with all software installed (recommended option)
- [pip](pip-installation)
- [conda](conda-installation)

The conda/pip installation method requires to install CORSIKA and sim_telarray separately, see [section below](#installation-of-corsika-and-sim_telarray).

### Container (docker)

OCI-compatible images are available for simtools users, developers, and for CORSIKA/sim_telarray from the [simtools package registry](https://github.com/orgs/gammasim/packages?repo_name=simtools).
These allows to skip all installation steps and run simtools applications directly.

See the [running simtools using containers](simtools_containers.md) page for more details.

### Pip Installation

simtools is available as a Python package from [PypPi](https://pypi.org/project/gammasimtools/).

To install, prepare a python environment, e.g.:

```console
mamba create --name simtools-prod python=3.11
mamba activate simtools-prod
```

Install simtools and its dependencies:

```console
pip install gammasimtools
```

### Conda Installation

Prepare and install a conda environment with the simtools package:

```console
conda env create -n gammasimtools
conda install gammasimtools --channel conda-forge
conda activate gammasimtools
```

## Installation of CORSIKA and sim_telarray

CORSIKA and sim_telarray are external tools to simtools and are required dependencies for many applications.
The installation requires some preparation, this is why it is recommended to use the Docker environment

For a non-Docker setup, follow the instruction provided by the CORSIKA/sim_telarray authors for installation.
CTAO users can download both packages from the [sim_telarray web page](https://www.mpi-hd.mpg.de/hfm/CTA/MC/Software/Testing/)
(CTAO password applies) and install the package with:

```console
tar -czf corsika7.7_simtelarray.tar.gz
./build_all prod6-sc qgs2 gsl
```

The environmental variable `$SIMTOOLS_SIMTEL_PATH` should point towards the CORSIKA/sim_telarray installation
(recommended to include it in the .env file with all other environment variables).

## Model Database Access

Simulation model parameters are stored in a MongoDB-type database.
Many simtools applications depend on access to this database.

:::{note}
Ask one of the developers for the credentials to access the database.
:::

Credentials for database access are passed on to simtools applications using environmental variables stored
in a file named `.env`.
Copy the template file [.env_template](https://github.com/gammasim/simtools/blob/main/.env_template)
to a new file named `.env` and update it with the credentials.

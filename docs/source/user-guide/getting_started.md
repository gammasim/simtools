# Getting Started

Using simtools requires installing and accessing its [main components](../components/index.md):
the [simtools package](#installation), the simulation software [CORSIKA and sim_telarray](#installation-of-corsika-and-sim_telarray), and the [simulation model database](model-database-access).

For development-related information, see [Getting Started as a Developer](../developer-guide/getting_started_as_developer.md).

## Installation

simtools can be installed using one of the following methods:

- Using a [container image](container-images) with all software pre-installed (**recommended**)
- Via [pip](pip-installation) or [conda](conda-installation). Requires manual compilation and installation of **CORSIKA** and **sim_telarray**. See the [section below](#installation-of-corsika-and-sim_telarray) for details. Note that the conda package might not always contain the latest simtools version.

## Container Images

OCI-compatible container images are available for simtools users and support both application and development use cases.  Any runtime such as [Docker](https://www.docker.com/products/docker-desktop), [Podman](https://podman.io/), or [Apptainer](https://apptainer.org/) can be used.
These images eliminate all manual installation steps and allow direct execution of simtools applications.

The most important types of images are list below; for a complete overview, see the [Container Images](container-images) documentation.

- **Simtools Production images** ([simtools-prod](https://github.com/gammasim/simtools/pkgs/container/simtools-sim-telarray-250903-corsika-78010-bernlohr-1.70-prod6-baseline-qgs3-no_opt)): Include CORSIKA, sim_telarray, and simtools applications. Variants are available with:
  - Different CORSIKA/sim_telarray versions
  - Compile options (e.g., `prod5`, `prod6`)
  - CPU optimizations (e.g., `avx2`, `avx512`, `generic`)
- **Simtools Development images** ([simtools-dev](https://github.com/gammasim/simtools/pkgs/container/simtools-dev)): Include all dependencies for simtools development, as well as CORSIKA and sim_telarray, but do not contain simtools itself.

Pre-built images are hosted on the [simtools package registry](https://github.com/orgs/gammasim/packages?repo_name=simtools). Authentication may be required; follow [GitHub's guide](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry) to configure access (`docker login`).


```{important}
Container images do not include the interaction tables required by CORSIKA. Follow the instructions in the [CORSIKA documentation](../components/corsika.md#corsika-interaction-tables) to download and install the interaction tables.
```

### Running a simtools Production Image

**Prerequisite**: Configure [simulation model database access](model-database-access).

Start an Interactive Container:

```bash
podman run --rm -it \
    --env-file .env \
    -v "$(pwd):/workdir/external" \
    ghcr.io/gammasim/simtools-240205-corsika-77500-bernlohr-1.67-prod6-baseline-qgs2-no_opt:latest \
    bash
```

Any simtools application can be run inside the container.

Run a simtools application:

```bash
podman run --rm -it \
    --env-file .env \
    -v "$(pwd):/workdir/external" \
    ghcr.io/gammasim/simtools-240205-corsika-77500-bernlohr-1.67-prod6-baseline-qgs2-no_opt:latest \
    simtools-convert-geo-coordinates-of-array-elements \
    --input ./simtools/tests/resources/telescope_positions-North-utm.ecsv \
    --export ground \
    --output_path /workdir/external/
```

### Pip Installation

simtools is available as a Python package from [PyPI](https://pypi.org/project/gammasimtools/).

To install, prepare a Python environment, e.g.:

```console
mamba create --name simtools-prod python=3.12
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
The installation requires some preparation; therefore, it is recommended to use the Docker environment.

For a non-Docker setup, follow the instructions provided by the CORSIKA/sim_telarray authors for installation.
CTAO users can download both packages from the [sim_telarray web page](https://www.mpi-hd.mpg.de/hfm/CTA/MC/Software/Testing/)
(CTAO password applies) and install the package with:

```console
tar -xzf corsika7.7_simtelarray.tar.gz
./build_all prod6-sc qgs2 gsl
```

The environmental variable `$SIMTOOLS_SIMTEL_PATH` should point towards the CORSIKA/sim_telarray installation
(recommended to include it in the .env file with all other environment variables).

## Model Database Access

Simulation model parameters are stored in the database.
Many simtools applications depend on access to this database.

:::{note}
Ask one of the developers for the credentials to access the database.
:::

Credentials for database access are passed on to simtools applications using environmental variables stored
in a file named `.env`, see the [Environment Variables](#environment-variables) section below.

(environment-variables)=

## Environment Variables

The environment variables listed below are used by simtools applications and defined by the user in a `.env` file to be placed in the working directory. Copy the template file [.env_template](https://github.com/gammasim/simtools/blob/main/.env_template) to a new file named `.env` and update it accordingly.

```console
# Hostname of the database server
SIMTOOLS_DB_SERVER=<hostname>
# Port on the database server
SIMTOOLS_DB_API_PORT=<integer>
# Username for database
SIMTOOLS_DB_API_USER=<username>
# Password for database
SIMTOOLS_DB_API_PW=<password>
SIMTOOLS_DB_API_AUTHENTICATION_DATABASE='admin'
# Name of the simulation model database
SIMTOOLS_DB_SIMULATION_MODEL='CTAO-Simulation-Model-v0-7-0'
# Path to the sim_telarray installation
SIMTOOLS_SIMTEL_PATH='/workdir/sim_telarray'
# User name of the user running the application
SIMTOOLS_USER_NAME='Max Mustermann'
# ORCID of the user running the application
SIMTOOLS_USER_ORCID='0000-1234-5678-0000'
```

```{note}
Any simtools application command-line argument can be set as an environment variable, see the [application configuration](applications.md#configuration) section.
```

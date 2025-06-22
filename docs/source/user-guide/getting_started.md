# Getting Started

Using simtools requires installing and accessing its [main components](../components/index.md):
the [simtools package](#installation), the simulation software  [CORSIKA and sim_telarray](#installation-of-corsika-and-sim_telarray), and the [simulation model database](model-database-access).

For development-related information, see [Getting Started as a Developer](../developer-guide/getting_started_as_developer.md).

## Installation

simtools can be installed using one of the following methods:

- Using a [container image](container-images) with all software pre-installed (**recommended**)
- Via [pip](pip-installation) or [conda](conda-installation). Requires manual compilation and installation of **CORSIKA** and **sim_telarray**. See the [section below](#installation-of-corsika-and-sim_telarray) for details.

## Container Images

OCI-compatible container images are available for simtools users and support both application and development use cases.  Any runtime such as [Docker](https://www.docker.com/products/docker-desktop), [Podman](https://podman.io/), or [Apptainer](https://apptainer.org/) can be used.
These images eliminate all manual installation steps and allow direct execution of simtools applications.

### Pre-built Images

- **Production images** (`simtools-prod`): Include CORSIKA, sim_telarray, and simtools applications. Variants are available with:
  - Different CORSIKA/sim_telarray versions
  - Compile options (e.g., `prod5`, `prod6`)
  - CPU optimizations (e.g., `avx2`, `avx512`, `no_opt`)
- **Development images** (`simtools-dev`): Include all dependencies for simtools development, as well as CORSIKA and sim_telarray, but do not contain simtools itself.

Pre-built images are hosted on the [simtools package registry](https://github.com/orgs/gammasim/packages?repo_name=simtools). Authentication may be required; follow [GitHub's guide](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry) to configure access (`docker login`).

### Running a simtools Production Image (`simtools-prod`)

```{warning}
todo - where is the env described?
```

**Prerequisite**: Configure simulation model database access (see the simtools documentation). An example `.env` file is available [here](https://github.com/gammasim/simtools/blob/main/.env_template).

Start an Interactive Container:

```bash
docker run --rm -it \
    --env-file .env \
    -v "$(pwd):/workdir/external" \
    ghcr.io/gammasim/simtools-prod-240205-corsika-77500-bernlohr-1.67-prod6-baseline-qgs2-no_opt:latest \
    bash
```

Run a simtools application:

```bash
docker run --rm -it \
    --env-file .env \
    -v "$(pwd):/workdir/external" \
    ghcr.io/gammasim/simtools-prod-240205-corsika-77500-bernlohr-1.67-prod6-baseline-qgs2-no_opt:latest \
    simtools-convert-geo-coordinates-of-array-elements \
    --input ./simtools/tests/resources/telescope_positions-North-utm.ecsv \
    --export ground \
    --output_path /workdir/external/
```

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

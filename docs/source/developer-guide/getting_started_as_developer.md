# Getting Started as Developer

The developer environment for simtools consists of the simtools packages, the CORSIKA/sim_telarray packages, and the model database.
The usage of the container for developers is strongly recommended and assumed to be the default environment for developers.
Note the correct setting of environment variables to connect to the model database (see also the [user's guide](../user-guide/getting_started.md))
and to point to the simulation software paths.

## Container Environment for Developers

Container images for developers are available from the [GitHub container registry](https://github.com/gammasim/simtools/pkgs/container/simtools-dev).
The images contain:

- an installation of CORSIKA and sim_telarray
- a python environment with all simtools dependencies (including the `dev` options of the project)
- g++ compiler and other build tools

The corresponding Docker file is [./docker/Dockerfile-dev](https://github.com/gammasim/simtools/blob/main/docker/Dockerfile-dev).

The simtools package is not installed in the container, but should be installed from source. This allows to develop using your editor of choice.

The following steps outline how to set up the development environment using containers

1. Clone simtools from GitHub into a directory `external/simtools`:

    ```bash
    # create a working directory
    mkdir external
    # clone simtools repository
    git clone https://github.com/gammasim/simtools.git external/simtools
    ```

2. Start up a container (the image will be downloaded, if it is not available in your environment):

    ```bash
    podman run --rm -it -v "$(pwd)/external:/workdir/external" \
        ghcr.io/gammasim/simtools-dev:latest \
        bash -c "source /workdir/env/bin/activate && cd /workdir/external/simtools && pip install -e . && bash"
    ```

    The container includes a CORSIKA and sim_telarray installation;
    the environmental variable `$SIMTOOLS_SIMTEL_PATH` and those for the database access are automatically set
    (if variables are set correctly in the `.env` file).

## Installation without Containers

Developers can install simtools directly from the [simtools GitHub repository](https://github.com/gammasim/simtools):

```console
git clone https://github.com/gammasim/simtools.git
cd simtools
```

Create a conda/mamba virtual environment with the simtools dependencies installed:

```console
mamba env create -f environment.yml
mamba activate simtools-dev
pip install -e .
```

To install the CORSIKA/sim_telarray packages, follow the installation instructions in the relevant manuals. A good guideline is also the step-by-step instructions outlined in the [CORSIKA/sim_telarray Docker file](https://github.com/gammasim/simtools/blob/main/docker/Dockerfile-prod-opt).

```{note}
The installation of CORSIKA/sim_telarray from source requires access to the
sim_telarray packages at MPIK (password applies).
```

## Testing your installation

Test the your installation by running the unit tests:

```console
pytest tests/unit_tests/
```

Test the simtools plus CORSIKA/sim_telarray installation by running the integration tests:

```console
pytest --no-cov tests/integration_tests/
```

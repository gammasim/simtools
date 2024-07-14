(devgetting-started)=

# Getting Started

The usage of simtools require the [installation of the simtools package](devinstallationfordevelopers), its dependencies,
and the setting of environment variables to connect to the model database (see also the [user's guide](../user-guide/getting_started.md)).

(devinstallationfordevelopers)=

## Installation

Developers install simtools directly from the [simtools GitHub repository](https://github.com/gammasim/simtools):

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

## Docker Environment for Developers

Docker containers are available for developers, see the [Docker file directory](https://github.com/gammasim/simtools/tree/main/docker).

Images are available from the [GitHub container registry](https://github.com/gammasim/simtools/pkgs/container/simtools-dev) for the latest simtools versions, for each pull request, and the current main branch.

The docker container has python packages, CORSIKA, and sim_telarray pre-installed.
Setting up a system to run simtools applications or tests should be a matter of minutes.

Install Docker and start the Docker application (see
[Docker installation page](https://docs.docker.com/engine/install/)).

Clone simtools from GitHub into a directory `external/simtools`:

```bash
# create a working directory
mkdir external
# clone simtools repository
git clone https://github.com/gammasim/simtools.git external/simtools
```

Start up a container (the image will be downloaded, if it is not available in your environment):

```bash
docker run --rm -it -v "$(pwd)/external:/workdir/external" \
    ghcr.io/gammasim/simtools-dev:latest \
    bash -c "source /workdir/env/bin/activate && cd /workdir/external/simtools && pip install -e . && bash"
```

The container includes a CORSIKA and sim_telarray installation;
the environmental variable `$SIMTOOLS_SIMTEL_PATH` and those for the database access are automatically set
(if variables are set correctly in the `.env` file).

(devtestinginstallation)=

## Testing your installation

Test the simtools installation the docker image by running the unit tests:

```console
pytest tests/unit_tests/
```

Test the simtools plus CORSIKA/sim_telarray installation by running the integration tests:

```console
pytest --no-cov tests/integration_tests/
```

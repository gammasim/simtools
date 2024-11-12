(getting-started)=

# Getting Started

The usage of simtools requires the installation of the [simtools package](installationforusers), its dependencies (mostly [CORISKA/sim_telarray](corsikasimtelarrayinstallation)),
and the setting of environment variables to [connect to the model database](modeldatabaseaccess).

(installationforusers)=

## Installation

There are four options to install simtools for users:

- [using conda](condainstallation)
- [using pip](pipinstallation)
- using Git and pip (this is the recommended method for developers) (see [Developer Installation](../developer-guide/getting_started.md))
- [using a docker container](dockerinstallation) with all software installed

All simtools applications are available as command-line tools.
Note the naming of the tool, starting with `simtools-` followed by the application name.
See the [applications](applications.md) section for more details.

Note to update the `.env` file with the credentials for database access (see [Model Database Access](databases.md)).

The conda/pip installation method requires to install CORSIKA/sim_telarray separately, see [CorsikaSimTelarrayInstallation].

(condainstallation)=

### Conda Installation

Prepare and install a conda environment with the simtools package:

```console
conda env create -n gammasimtools
conda install gammasimtools --channel conda-forge
conda activate gammasimtools
```

(pipinstallation)=

### Pip Installation

Prepare a python environment (in this example for python version 3.11):

```console
mamba create --name simtools-prod python=3.11
mamba activate simtools-prod
```

Use pip to install simtools and its dependencies:

```console
pip install gammasimtools
```

The pip installation method requires to [install CORSIKA/sim_telarray](corsikasimtelarrayinstallation) separately.

(dockerinstallation)=

### Container (docker)

OCI-compatible images are available for simtools users, developers, and for CORSIKA/sim_telarray from the [simtools package registry](https://github.com/orgs/gammasim/packages?repo_name=simtools).
These allows to skip all installation steps and run simtools applications directly.

See the [Docker description](docker_files.md) for more details.

(corsikasimtelarrayinstallation)=

## Installation of CORSIKA and sim_telarray

CORSIKA and sim_telarray are external tools to simtools and are required dependencies for many applications.
Recommended is to use the Docker environment, see description in [Docker Environment for Developers](docker_files.md).

For a non-Docker setup, follow the instruction provided by the CORSIKA/sim_telarray authors for installation.
CTAO users can download both packages from the [sim_telarray web page](https://www.mpi-hd.mpg.de/hfm/CTA/MC/Software/Testing/)
(CTAO password applies) and install the package with:

```console
tar -czf corsika7.7_simtelarray.tar.gz
./build_all prod6-sc qgs2 gsl
```

The environmental variable `$SIMTOOLS_SIMTEL_PATH` should point towards the CORSIKA/sim_telarray installation
(recommended to include it in the .env file with all other environment variables).

(modeldatabaseaccess)=

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

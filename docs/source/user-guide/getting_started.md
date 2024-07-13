(getting-started)=

# Getting Started

The usage of simtools require the installation of the simtools package, its dependencies,
and the setting of environment variables to connect to the model database.

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

(installationforusers)=

## Installation

:::{warning}
simtools is under rapid development and not ready for production use.
The following setup is recommended for users who want to test the software.
:::

There are three options to install simtools for users:

- using pip
- using git and pip
- download a docker container with all software installed

All simtools applications are available as command-line tools.
Note the naming of the tool, starting with `simtools-` followed by the application name.
See {ref}`Applications` for more details.

Note to update the `.env` file with the credentials for database access (see [Model Database Access]).

(pipinstallation)=

### Pip Installation

:::{warning}
The pip-installation of simtools provides limited functionality only
and is not as well tests as the conda/mamba installation.
:::

Prepare a python environment (in this example for python version 3.11):

```console
mamba create --name simtools-prod python=3.11
mamba activate simtools-prod
```

Use pip to install simtools and its dependencies:

```console
pip install gammasimtools
```

The pip installation method requires to install CORSIKA/sim_telarray separately, see [CorsikaSimTelarrayInstallation].

(gitinstallation)=

### Git Installation

Install simtools directly from the GitHub repository:

```console
git clone https://github.com/gammasim/simtools.git
cd simtools
pip install .
```

The git installation method requires to install CORSIKA/sim_telarray separately, see [CorsikaSimTelarrayInstallation].

(dockerinstallation)=

### Container (docker)

OCI-compatible images are available for simtools users, developers, and for CORSIKA/sim_telarray from the [simtools package registry](https://github.com/orgs/gammasim/packages?repo_name=simtools).
These allows to skip all installation steps and run simtools applications directly.

See the [Docker description](docker_files.md) for more details.

## Installation of CORSIKA and sim_telarray

CORSIKA and sim_telarray are external tools to simtools and are required dependencies for many applications.
Recommended is to use the Docker environment, see description in [Docker Environment for Developers](docker_files.md).

For a non-Docker setup, follow the instruction provided by the CORSIKA/sim_telarray authors for installation.
CTAO users can download both packages from the [sim_telarray web page](https://www.mpi-hd.mpg.de/hfm/CTA/MC/Software/Testing/)
(CTAO password applies) and install the package with:

```console
tar -czf corsika7.7_simtelarray.tar.gz
./build_all prod5 qgs2 gsl
```

The environmental variable `$SIMTOOLS_SIMTEL_PATH` should point towards the CORSIKA/sim_telarray installation
(recommended to include it in the .env file with all other environment variables).

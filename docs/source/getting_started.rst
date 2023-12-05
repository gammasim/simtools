.. _Getting_Started:

Getting Started
***************

The usage of simtools require the installation of the simtools package, its dependencies,
and the setting of environment variables to connect to the model database.

Model Database Access
=====================

Simulation model parameters are stored in a MongoDB-type database.
Many simtools applications depend on access to this database.

.. note::

    Ask one of the developers for the credentials to access the database.

Credentials for database access are passed on to simtools applications using environmental variables stored
in a file named ``.env``.
Copy the template file `.env_template <https://github.com/gammasim/simtools/blob/main/.env_template>`_
to a new file named ``.env`` and update it with the credentials.

.. _InstallationForUsers:

Installation for Users
======================

.. warning::

    simtools is under rapid development and not ready for production use.
    The following setup is recommended for users who want to test the software.

There are three options to install simtools for users:

* using pip
* using git and pip
* download a docker container with all software installed

All simtools applications are available as command line tools.
Note the naming of the tool, starting with ``simtools-`` followed by the application name.
See :ref:`Applications` for more details.

Note to update the ``.env`` file with the credentials for database access (see `Model Database Access`_).

.. _PipInstallation:

Pip Installation
----------------

.. warning::

    The pip-installation of simtools provides limited functionality only
    and is not as well tests as the conda/mamba installation.

Prepare a python environment (in this example for python version 3.11):

.. code-block:: console

    $ mamba create --name simtools-prod python=3.11
    $ mamba activate simtools-prod

Use pip to install simtools and its dependencies:

.. code-block:: console

    $ pip install gammasimtools

The pip installation method requires to install CORSIKA/sim_telarray separately, see `CorsikaSimTelarrayInstallation`_.

.. _GitInstallation:

Git Installation
----------------

Install simtools directly from the GitHub repository:

.. code-block:: console

    $ git clone https://github.com/gammasim/simtools.git
    $ cd simtools
    $ pip install .

The git installation method requires to install CORSIKA/sim_telarray separately, see `CorsikaSimTelarrayInstallation`_.

.. _DockerInstallation:

Docker Installation
-------------------

The docker container ``simtools-prod`` includes all software required to run simtools applications:

* corsika and sim\_telarray
* python packages required by simtools
* simtools

To run bash in the `simtools-prod container  <https://github.com/gammasim/simtools/pkgs/container/simtools-prod>`_:

.. code-block:: console

    docker run --rm -it --env-file .env \
        -v "$(pwd):/workdir/external" \
        ghcr.io/gammasim/simtools-prod:latest bash

In the container, simtools applications are installed and can be called directly (e.g., ``simtools-print-array-elements -h``).
This example uses the docker syntax to mount your local directory.

The following example runs an application inside the container and writes the output into a directory of the local files system,

.. code-block:: console

    docker run --rm -it --env-file .env \
        -v "$(pwd):/workdir/external" \
        ghcr.io/gammasim/simtools-prod:latest \
        simtools-print-array-elements \
        --array_element_list ./simtools/tests/resources/telescope_positions-North-utm.ecsv \
        --export corsika --use_corsika_telescope_height \
        --output_path /workdir/external/

.. _InstallationForDevelopers:

Installation for Developers
===========================

Developers install simtools directly from the GitHub repository:

.. code-block:: console

    $ git clone https://github.com/gammasim/simtools.git
    $ cd simtools

Create a conda/mamba virtual environment with the simtools dependencies installed:

.. code-block:: console

    $ mamba env create -f environment.yml
    $ mamba activate simtools-dev
    $ pip install -e .

.. _CorsikaSimTelarrayInstallation:

Installation of CORSIKA and sim_telarray
========================================

CORSIKA and sim_telarray are external tools to simtools and are required dependencies for many applications.
Recommended is to use the Docker environment, see description in  `Docker Environment for Developers`_.

For a non-Docker setup, follow the instruction provided by the CORSIKA/sim_telarray authors for installation.
CTA users can download both packages from the `sim_telarray webpage <https://www.mpi-hd.mpg.de/hfm/CTA/MC/Software/Testing/>`_
(CTA password applies) and install the package with:

.. code-block:: console

    $ tar -czf corsika7.7_simtelarray.tar.gz
    $ ./build_all prod5 qgs2 gsl

The environmental variable ``$SIM_TELPATH`` should point towards the CORSIKA/sim_telarray installation
(recommended to include it in the \.env file with all other environment variables).


Test your complete installation following the instructions in :ref:`this section <TestingInstallation>`.

Docker Environment for Developers
=================================

Docker containers are available for developers, see the `Docker file directory <https://github.com/gammasim/simtools/tree/main/docker>`_.

Images are available from the `GitHub container registry <https://github.com/gammasim/simtools/pkgs/container/simtools-dev>`_ for the latest simtools versions, for each pull request, and the current main branch.

The docker container has python packages, CORSIKA, and sim_telarray pre-installed.
Setting up a system to run simtools applications or tests should be a matter of minutes.

Install Docker and start the Docker application (see
`Docker installation page <https://docs.docker.com/engine/install/>`_). Other container systems like
Apptainer, Singularity, Buildah/Podman, should work, but are not thoroughly tested.

Clone simtools from GitHub into a directory ``external/simtools``:

.. code-block::

    # create a working directory
    mkdir external
    # clone simtools repository
    git clone https://github.com/gammasim/simtools.git external/simtools

Start up a container (the image will be downloaded, if it is not available in your environment):

.. code-block::

    docker run --rm -it -v "$(pwd)/external:/workdir/external" \
        ghcr.io/gammasim/simtools-dev:latest \
        bash -c "source /workdir/env/bin/activate && cd /workdir/external/simtools && pip install -e . && bash"

The container includes a CORSIKA and sim_telarray installation;
the environmental variable ``$SIM_TELPATH`` and those for the database access are automatically set
(if variables are set correctly in the \.env` file).

Test your installation following the instructions in :ref:`this section <TestingInstallation>`.

.. _TestingInstallation:

Testing your installation
=========================

Test the simtools installation the docker image by running the unit tests:

.. code-block:: console

    $ pytest tests/unit_tests/

Test the simtools plus CORSIKA/sim_telarray installation by running the integration tests:

.. code-block:: console

    $ pytest tests/integration_tests/

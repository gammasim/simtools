.. _Getting_Started:

Getting Started
***************

The usage of simtools require the installation of the simtools package, its dependencies,
and the setting of environment variables to connect to the model database.

Model Database Access
---------------------

Simulation model parameters are stored in a MongoDB-type database.
Many simtools applications depend on access to this database; ask one of the developers for the credentials.

Credentials for database access are passed on to simtools applications using environmental variables.
For database access, copy \
`set_DB_environ_template.sh <https://github.com/gammasim/simtools/blob/main/set_DB_environ_template.sh>`_ to a new file named ``set_DB_environ.sh``, and update it with the credentials:

.. code-block::

    export DB_API_USER=<db_user_name>
    export DB_API_PW=<db_password>
    export DB_API_PORT=<db_port>
    export DB_SERVER=<db_server>

See below for the usage of this script.

.. _InstallationForUsers:

Installation for Users
----------------------

simtools is under rapid development and not ready for production use.
The following setup is recommended for users who want to test the software.

.. warning::

    The pip-installation of simtools provides limited functionality only
    and is not as well tests as the conda/mamba installation.

Install a simple python environment:

.. code-block:: console

    $ mamba create --name simtools-prod python=3.9
    $ mamba activate simtools-prod

Use pip to install simtools and its dependencies:

.. code-block:: console

    $ pip install gammasimtools

Alternatively, install simtools directly from the GitHub repository:

.. code-block:: console

    $ git clone https://github.com/gammasim/simtools.git
    $ cd simtools
    $ pip install .

Source the ``set_DB_environ.sh`` script in order to run the applications (see `Model Database Access`_):

.. code-block:: console

    $ source set_DB_environ.sh


All simtools applications are now available as command line tools.
Note the naming of the tool, starting with ``simtools-`` followed by the application name.
See :ref:`Applications` for more details.

.. _InstallationForDevelopers:

Installation for Developers
---------------------------

Developers install simtools directly from the GitHub repository:

.. code-block:: console

    $ git clone https://github.com/gammasim/simtools.git
    $ cd simtools

Create a conda/mamba virtual environment with the simtools dependencies installed:

.. code-block:: console

    $ mamba env create -f environment.yml
    $ mamba activate simtools-dev
    $ pip install -e .

CORSIKA and sim_telarray are external tools to simtools and are used by several simtools applications.
Follow the instruction provided by the CORSIKA/sim_telarray authors for installation.
CTA users can download both packages from the `sim_telarray webpage <https://www.mpi-hd.mpg.de/hfm/CTA/MC/Software/Testing/>`_ (password applies) and install the package with e.g.:

.. code-block:: console

    $ tar -czf corsika7.7_simtelarray.tar.gz
    $ ./build_all prod5 qgs2 gsl

Source the ``set_DB_environ.sh`` script (see `Model Database Access`_) to activate set the environmental variables for the DB access:

.. code-block:: console

    $ source set_DB_environ.sh

The environmental variable ``$SIM_TELPATH`` should point towards the CORSIKA/sim_telarray installation.

Test your installation by running the unit tests:

.. code-block:: console

    $ pytest tests/unit_tests/

Docker Environment for Developers
---------------------------------

A docker container is made available for developers, see the `Docker file directory <https://github.com/gammasim/simtools/tree/main/docker>`_.

Images are available from the `GitHub container registry <https://github.com/gammasim/simtools/pkgs/container/simtools-dev>`_ for the latest simtools versions, for each pull request, and the current main branch.

The docker container has python packages, CORSIKA, and sim_telarray pre-installed.
Setting up a system to run simtools applications or tests should be a matter of minutes.

Install Docker and start the Docker application (see
`Docker installation page <https://docs.docker.com/engine/install/>`_). Other container systems like
Apptainer, Singularity, Buildah/Podman, etc should work, but are not thoroughly tested.

Clone simtools from GitHub into ``external/simtools``:

.. code-block::

    # create a working directory
    mkdir external
    # clone simtools repository
    git clone https://github.com/gammasim/simtools.git external/simtools

Start up a container (the image will we downloaded, if it is not available in your environment):

.. code-block::

    docker run --rm -it -v "$(pwd)/external:/workdir/external" docker pull ghcr.io/gammasim/simtools-dev:latest bash -c "$(cat ./entrypoint.sh) && bash"

The entry script of the container will source the ``set_DB_environ.sh`` script and set the DB access parameters (see `Model Database Access`_).
The container includes a CORSIKA and sim_telarray installation; the environmental variable ``$SIM_TELPATH`` is set.

Test your installation using the docker image by running the unit tests:

.. code-block:: console

    $ pytest tests/unit_tests/

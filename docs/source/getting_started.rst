.. _Getting_Started:

Getting Started
***************

Model Database Access
---------------------

Simulation model parameters are stored in a MongoDB-type database.
Many simtools applications depend on access to this database; ask one of the developers for the credentials.

Credentials for database access are passed on to simtools applications using environmental variables.
For database access, copy \
`set_DB_environ_template.sh <https://github.com/gammasim/simtools/blob/master/set_DB_environ_template.sh>`_ to a new file named ``set_DB_environ.sh``, and update it with the credentials:

.. code-block::

    export DB_API_USER=<db_user_name>
    export DB_API_PW=<db_password>
    export DB_API_PORT=<db_port>
    export DB_SERVER=<db_server>

See below for the usage of this script.

Installation for Users
----------------------

simtools is under rapid development and not ready for production use.
It will be made available in future using the conda packaging system.
For now, expert users should follow the installation procedures for developers.


Installation for Developers
---------------------------

Developers install simtools directly from the GitHub repository:

++++++++++++++++++++
Clone simtools
++++++++++++++++++++

.. code-block:: console

    $ git clone https://github.com/gammasim/simtools.git

++++++++++++++++++++
Install dependencies
++++++++++++++++++++

Create a conda virtual environment with the simtools dependencies installed:

.. code-block:: console

    $ conda env create -f environment.yml

    $ conda activate simtools-dev

    $ pip install -e .

++++++++++++++++++++++++++++++++
Install CORSIKA and sim_telarray
++++++++++++++++++++++++++++++++

CORSIKA and sim_telarray are external tools to simtools and are used by several gammasim-tools applications.
Follow the instruction provided by the CORSIKA/sim_telarray authors for installation.
CTA users can download both packages from `here <https://www.mpi-hd.mpg.de/hfm/CTA/MC/Software/Testing/>`_ and install the package with e.g.:

.. code-block:: console

    tar -czf corsika7.7_simtelarray.tar.gz
    ./build_all prod5 qgs2 gsl


+++++++++++++++++++++++++++
Set environmental variables
+++++++++++++++++++++++++++

Source the ``set_DB_environ.sh`` script (see `Model Database Access`_) to activate set the environmental variables for the DB access:

.. code-block:: console

    $ source set_DB_environ.sh

The environmental variable ``$SIM_TELPATH`` should point towards the CORSIKA/sim_telarray installation.

+++++++++++++++++
Test installation
+++++++++++++++++

Test your installation by running the unit tests:

.. code-block:: console

    $ pytest tests/unit_tests/

Docker Environment for Developers
---------------------------------

A docker container is made available for developers, see the
`simtools container repository <https://github.com/gammasim/containers/tree/main/dev>`_ for the Docker files.
Images are uploaded to `package section <https://github.com/orgs/gammasim/packages?repo_name=containershttps://github.com/orgs/gammasim/packages?repo_name=containers>`_ of this repository (at this point a private container repository; ask the simtools developers for access).

The container has python packages, CORSIKA, and sim_telarray pre-installed.
Setting up a system to run simtools applications or tests should be a matter of minutes.

+++++++++++++++++++
Docker Installation
+++++++++++++++++++

Install Docker and start the Docker application (see
`Docker installation page <https://docs.docker.com/engine/install/>`_). Other container systems like
Apptainer, Singularity, Buildah/Podman, etc should work, but are not thoroughly tested.

++++++++++++++++++++
Clone simtools
++++++++++++++++++++

Clone simtools from GitHub into ``external/gammasim-tools``:

.. code-block::

    # create a working directory
    mkdir external
    # clone simtools repository
    git clone https://github.com/gammasim/simtools.git external/gammasim-tools

+++++++++++++++++++++
Spin-up the container
+++++++++++++++++++++

Start up a container (the image will we downloaded, if it is not available in your environment):

.. code-block::

    docker run --rm -it -v "$(pwd)/external:/workdir/external" ghcr.io/gammasim/containers/simtools-dev:v0.3.0-dev1 bash -c "$(cat ./entrypoint.sh) && bash"

The entry script of the container will source the ``set_DB_environ.sh`` script and set the DB access parameters (see `Model Database Access`_).
The container includes a CORSIKA and sim_telarray installation; the environmental variable ``$SIM_TELPATH`` is set.

+++++++++++++++++
Test installation
+++++++++++++++++

Test your installation by running the unit tests:

.. code-block:: console

    $ pytest tests/unit_tests/

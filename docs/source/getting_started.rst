.. _Getting_Started:

Getting Started
***************

A conda (or pip) installation is still not possible. Meanwhile,
you need to clone (or fork) the repository.

.. code-block:: console

    $ git clone https://github.com/gammasim/gammasim-tools.git


Within the repository, you will find a conda environment file.
You can create and activate the environment by:

.. code-block:: console

    $ conda env create -f environment.yml

    $ conda activate gammasim-tools-dev

    $ pip install -e .


Each time you want to use the package, just activate the conda environment:

.. code-block:: console

    $ conda activate gammasim-tools-dev

If you need access to the DB, ask one of the developers for the credentials, copy \
set_env_db_template.sh to a new file named set_env_db.sh, update it with the credentials and \
source it to activate the environmental variables:

.. code-block:: console

    $ source set_env_db.sh

You are all set now.

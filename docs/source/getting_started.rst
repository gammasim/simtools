.. _Getting_Started:

Getting Started
***************

A conda (or pip) installation is still not posssible. Meanwhile,
you need to clone (or fork) the repository.

.. code-block:: console

    $ git clone https://github.com/gammasim/gammasim-tools.git


Within the repository, you will find a conda environment file.
You can create and activate the environment by:

.. code-block:: console

    $ conda env create -f environment.yml

    $ conda activate gammasim-tools-dev


You will also find in the repository a bash script to add the simtools library
to your python path. 

.. code-block:: console

    $ source set_simtools.sh

You are all set now =)

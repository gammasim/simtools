.. _inspect_file:

simtools-inspect-file
=====================

.. automodule:: inspect_file
   :members:

To print the content of a root dataset from an HDF5 simulation product, select it by name:

.. code-block:: console

   simtools-inspect-file output.reduced_event_data.hdf5 \
       --show_entry SIMULATION_METADATA

JSON metadata entries are formatted as JSON. Compound table datasets such as ``SHOWERS`` are
printed using the existing HDF5 table reader, with all columns shown and rows limited by
``--max_entries``.

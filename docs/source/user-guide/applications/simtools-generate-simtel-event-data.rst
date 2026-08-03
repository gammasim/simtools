
simtools-generate-simtel-event-data
===================================

The generated HDF5 file embeds ``METADATA`` and ``SIMULATION_METADATA`` JSON datasets alongside the
event tables. The latter records input run headers and sim_telarray metadata; resolved model
records are included when the workflow has them, without embedding model-file contents.

.. automodule:: generate_simtel_event_data
   :members:
   :exclude-members: main

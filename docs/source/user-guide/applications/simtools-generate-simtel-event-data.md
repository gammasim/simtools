# simtools-generate-simtel-event-data

```{eval-rst}
.. automodule:: simtools.applications.generate_simtel_event_data
   :members:
   :exclude-members: main
```

```{eval-rst}
Processes sim_telarray output files (typically of type '.simtel.zst') and creates
reduced datasets containing shower information, array-level parameters, and data about
triggered telescopes.

The output consists of an HDF5 file containing the following tables:

**FILE_INFO**

+-------------------+---------+-----------------------------------------------+
| Field             | Type    | Description                                   |
+===================+=========+===============================================+
| file_name         | string  | Name of the file                              |
+-------------------+---------+-----------------------------------------------+
| file_id           | int64   | Internal unique identifier for the file       |
+-------------------+---------+-----------------------------------------------+
| particle_id       | int64   | PDG particle ID (e.g., 14 for proton)         |
+-------------------+---------+-----------------------------------------------+
| energy_min        | float32 | Minimum simulated energy (TeV)                |
+-------------------+---------+-----------------------------------------------+
| energy_max        | float32 | Maximum simulated energy (TeV)                |
+-------------------+---------+-----------------------------------------------+
| viewcone_min      | float32 | Min viewcone angle (deg)                      |
+-------------------+---------+-----------------------------------------------+
| viewcone_max      | float32 | Max viewcone angle (deg)                      |
+-------------------+---------+-----------------------------------------------+
| core_scatter_min  | float32 | Min core scatter radius (m)                   |
+-------------------+---------+-----------------------------------------------+
| core_scatter_max  | float32 | Max core scatter radius (m)                   |
+-------------------+---------+-----------------------------------------------+
| zenith            | float32 | Zenith angle (deg)                            |
+-------------------+---------+-----------------------------------------------+
| azimuth           | float32 | Azimuth angle (deg)                           |
+-------------------+---------+-----------------------------------------------+
| nsb_level         | float64 | Night sky background level (factor to dark)   |
+-------------------+---------+-----------------------------------------------+

**SHOWERS**

+------------------+---------+-----------------------------------------------+
| Field            | Type    | Description                                   |
+==================+=========+===============================================+
| shower_id        | int64   | Shower identifier                             |
+------------------+---------+-----------------------------------------------+
| event_id         | int64   | Event identifier (depends on reuse of showers)|
+------------------+---------+-----------------------------------------------+
| file_id          | int64   | Internal unique identifier for the file       |
+------------------+---------+-----------------------------------------------+
| simulated_energy | float64 | Simulated primary energy (TeV)                |
+------------------+---------+-----------------------------------------------+
| x_core           | float64 | Shower core X position on ground (m)          |
+------------------+---------+-----------------------------------------------+
| y_core           | float64 | Shower core Y position on ground (m)          |
+------------------+---------+-----------------------------------------------+
| shower_azimuth   | float64 | Direction of shower azimuth (deg)             |
+------------------+---------+-----------------------------------------------+
| shower_altitude  | float64 | Direction of shower altitude (deg)            |
+------------------+---------+-----------------------------------------------+
| area_weight      | float64 | Weighting factor for sampling area            |
+------------------+---------+-----------------------------------------------+

**TRIGGERS**

+-----------------+---------+-----------------------------------------------+
| Field           | Type    | Description                                   |
+=================+=========+===============================================+
| shower_id       | int64   | Shower identifier                             |
+-----------------+---------+-----------------------------------------------+
| event_id        | int64   | Event identifier (depends on reuse of showers)|
+-----------------+---------+-----------------------------------------------+
| file_id         | int64   | Internal unique identifier for the file       |
+-----------------+---------+-----------------------------------------------+
| array_altitude  | float64 | Altitude of array pointing direction (deg)    |
+-----------------+---------+-----------------------------------------------+
| array_azimuth   | float64 | Azimuth of array pointing direction (deg)     |
+-----------------+---------+-----------------------------------------------+
| telescope_list  | string  | Comma-separated list of triggered telescopes  |
+-----------------+---------+-----------------------------------------------+

**Command line arguments**

prefix (str, required)
    Path prefix for the input files.
output_file (str, required)
    Output file path.
max_files (int, optional)
    Maximum number of input files to process. By default, process all input files.
print_dataset_information (int, optional, default=0)
    Print information about the datasets in the generated reduced event dataset.

**Example**

Generate a reduced dataset from input files and save the result.

.. code-block:: console

    simtools-production-extract-mc-event-data \\
    simtools-generate-simtel-event-data \\
        --simtel_file 'path/to/input_files/gamma_*dark*.simtel.zst' \\
        --output_file output_file.hdf5 \\
        --max_files 50 \\
        --print_dataset_information 10


To read a reduced event data file, use the following command reading on of the test files:

.. code-block:: console

    import h5py

    test_file = "tests/resources/proton_za20deg_azm000deg_North_alpha_6.0.0_reduced_event_data.hdf5"

    with h5py.File(test_file, "r") as f:
        triggers = f["/TRIGGERS"]
        for row in triggers:
            print({name: row[name] for name in row.dtype.names})
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: generate_simtel_event_data
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: generate_simtel_event_data_hdf5.yml
```

# simtools-generate-corsika-histograms

```{eval-rst}
.. automodule:: simtools.applications.generate_corsika_histograms
   :members:
   :exclude-members: main
```

```{eval-rst}
The Cherenkov photons (from observation level) are read from a CORSIKA IACT
output file(s) provided as input.

The following 2D histograms are generated:

    - Density of Cherenkov photons on the ground
    - Incoming direction (directive cosines) of the Cherenkov photons
    - Time of arrival (ns) vs altitude of production (km)

The following 1D histograms are generated:

    - Wavelength distribution of Cherenkov photons
    - Time of arrival (ns) distribution of Cherenkov photons
    - Altitude of production (km) distribution of Cherenkov photons
    - Lateral distribution of Cherenkov photons (distance from shower core in m)

**Command line arguments**
input_files (str, required)
    The name(s) of the CORSIKA IACT file(s) resulted from the CORSIKA simulation.

pdf_file_name (str, optional)
    The name of the output pdf file to save the histograms. If not provided,
    the histograms are only shown on screen.

file_labels (str, optional)
    Labels for the input files (in the same order as input_files). If not provided,
    the file names are used as labels.

**Example**
Fill and plot histograms for a test IACT file:

 .. code-block:: console

    simtools-generate-corsika-histograms --input_files /workdir/external/simtools/\\
    tests/resources/tel_output_10GeV-2-gamma-20deg-CTAO-South.corsikaio \\
        --pdf_file_name test.pdf

Fill and plot histograms for several files:

 .. code-block:: console

    simtools-generate-corsika-histograms --input_files file1 file 2 \\
        --file_lablels label1 label2 \\
        --pdf_file_name test.pdf

**Notes**
The typical use case of this application is to generate lateral photon density distribution
to compare different CORSIKA simulation settings or different CORSIKA versions. The following
steps are recommended:

    - generate a 'star'-like array of telescopes with the 'simtools-generate-regular-arrays'
      application. There should be a sufficient number of telescopes (e.g. 50 or more) in the
      layout with non-overlapping telescope definitions

    - run CORSIKA simulations with the desired settings using this telescope layout (use the
      'overwrite_model_parameters' option to point to the generated layout simulation model
      change file (in the format given by 'simulation_models_info.schema.yml').

    - run this application to generate the histograms for the produced CORSIKA IACT output
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: generate_corsika_histograms
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: generate_corsika_histograms_plot.yml
```

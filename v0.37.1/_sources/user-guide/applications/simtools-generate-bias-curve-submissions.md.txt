# simtools-generate-bias-curve-submissions

```{eval-rst}
.. automodule:: simtools.applications.generate_bias_curve_submissions
   :members:
   :exclude-members: main
```

```{eval-rst}
This application always prepares both curves:

- NSB curve
- Proton curve with proton primary

For each curve, the application creates a production grid and
parameter-scan grid:

- ``base_grid.ecsv``
- ``scan_config.yaml``
- ``scan_grid.ecsv``

The resulting scan grids can be consumed by a backend-specific submission generator,
for example ``simtools-simulate-prod-htcondor-generator``.

**Command line arguments**

site (str, required)
    Observation site (e.g., North, South).
model_version (str, required)
    Simulation model version.
array_layout_name (str, required)
    Single-telescope array layout name for simulations.
simulation_software (str)
    Simulation software (default: corsika_sim_telarray).
azimuth_angle (float, required)
    Azimuth angle in degrees.
zenith_angle (float, required)
    Zenith angle in degrees.
showers_per_run (int, required)
    Number of showers per run.
core_scatter (str, required)
    Core scatter (e.g., "20 1900 m").
view_cone (str, required)
    View cone (e.g., "0 deg 5 deg").
number_of_runs (int, required)
    Number of runs.
corsika_le_interaction (str)
    CORSIKA low-energy interaction model (default: urqmd).
corsika_he_interaction (str)
    CORSIKA high-energy interaction model (default: epos).
corsika_hadronic_transition_energy (Quantity)
    Transition energy between low- and high-energy hadronic models. If omitted, use the CORSIKA
    build default.
nsb_energy_range (str)
    NSB gamma energy range (default: "20 MeV 25 MeV").
proton_energy_range (str)
    Proton energy range (default: "2 GeV 2000 GeV").
nsb_scaling_factor (float)
    NSB scaling factor used for both curves (default: 2).
trigger_thresholds (float, int, float, optional)
    Three values defining the trigger-threshold scan used for both curves:
    minimum threshold, number of thresholds, and step size. Trigger-dependent
    defaults are used when omitted.
output_path (Path)
    Root output directory; nsb/ and proton/ sub-dirs are created inside it
    (provided by framework, default: ./simtools-output/).

**Example**

.. code-block:: console

    simtools-generate-bias-curve-submissions \
        --site North \
        --model_version 7.0.0 \
        --array_layout_name LSTN-01 \
        --azimuth_angle 0.0 \
        --zenith_angle 20.0 \
        --showers_per_run 10000 \
        --core_scatter "20 1900 m" \
        --view_cone "0 deg 5 deg" \
        --number_of_runs 10 \
        --nsb_energy_range "20 MeV 25 MeV" \
        --proton_energy_range "2 GeV 2000 GeV" \
        --nsb_scaling_factor 2 \
        --trigger_thresholds 220 3 10 \
        --output_path ./bias_curves

Submit files can be generated explicitly for a chosen backend, for
example:

.. code-block:: console

    simtools-simulate-prod-htcondor-generator \
        --job_grid_file ./bias_curves/nsb/scan_grid.ecsv \
        --output_path ./bias_curves/nsb/htcondor_submit \
        --apptainer_image /path/to/image.sif \
        --label nsb
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: generate_bias_curve_submissions
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: generate_bias_curve_submissions_lstn-01.yml
```

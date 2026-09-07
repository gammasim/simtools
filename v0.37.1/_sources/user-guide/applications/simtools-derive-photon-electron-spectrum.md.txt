# simtools-derive-photon-electron-spectrum

```{eval-rst}
.. automodule:: simtools.applications.derive_photon_electron_spectrum
   :members:
   :exclude-members: main
```

```{eval-rst}
Normalizes singe-p.e. amplitude distribution to mean amplitude of 1.0,
as required by sim_telarray. Allows to fold in afterpulse distribution
to prompt a spectrum. Uses the sim_telarray tool 'norm_spe' to normalize
the spectra.

Input files can be in ecsv format (preferred) or in the sim_telarray legacy format.

Two output files with identical data are written to the output directory:

- 'output_file'.ecsv: Single photon electron spectrum in ecsv format (data and metadata).
- 'output_file'.dat: Single photon electron spectrum in sim_telarray format.

**Example**

.. code-block:: console

    simtools-derive-photon-electron-spectrum \\
        --input_spectrum spectrum_photon_electron.ecsv \\
        --afterpulse_spectrum spectrum_afterpulse.ecsv \\
        --step_size 0.02 \\
        --max_amplitude 42.0 \\
        --use_norm_spe \\
        --output_path ./tests/output \\
        --output_file spectrum_photon_electron_afterpulse.ecsv

For an example of how to plot the single photon electron spectrum, see the
integration test 'tests/integration_tests/config/plot_tabular_data_for_single_pe_data.yml'.
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: derive_photon_electron_spectrum
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: derive_photon_electron_spectrum_lst_ecsv.yml
```

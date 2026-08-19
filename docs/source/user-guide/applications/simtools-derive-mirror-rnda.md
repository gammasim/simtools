# simtools-derive-mirror-rnda

```{eval-rst}
.. automodule:: simtools.applications.derive_mirror_rnda
   :members:
   :exclude-members: main
```

```{eval-rst}
**Description**

This application derives the value of the simulation model parameter
*mirror_reflection_random_angle* using measurements of a PSF containment diameter
and focal length of individual mirror panels.

The optimization uses percentage difference as the metric::

    pct_diff = 100 * (simulated_psf - measured_psf) / measured_psf

Each mirror is optimized individually, and the final RNDA is the average of all
per-mirror optimized values.

**Command line arguments**

site (str, required)
    North or South.
telescope (str, required)
    Telescope name (e.g. LSTN-01, SSTS-25).
model_version (str, optional)
    Model version.
data (str, required)
    ECSV file with PSF diameter (mm) per mirror.
    Accepted column names: psf_opt, psf, or d80.
fraction (float, optional)
    PSF containment fraction for diameter calculation (e.g. 0.8 for D80, 0.95 for D95).
    Default: 0.8.
threshold (float, optional)
    Convergence threshold for percentage difference (e.g. 0.05 for 5%).
    Default: 0.05.
learning_rate (float, optional)
    Learning rate for gradient descent. Default: 0.001.
test (optional)
    Only optimize a small number of mirrors.
max_workers (int, optional)
    Number of parallel worker processes to use. Default: 0 (auto chooses maximum).
number_of_mirrors_to_test (int, optional)
    Number of mirrors to optimize when --test is used. Default: 10.
profile_serial (optional)
    Run optimization in a single process (no process pool). Useful for profiling.
psf_hist (str, optional)
    If activated, write a histogram comparing measured vs simulated PSF diameter distributions.
cleanup (optional)
    Remove intermediate files (patterns: ``*.log``, ``*.lis*``, ``*.dat``)
    from output.

**Example**

.. code-block:: console

    simtools-derive-mirror-rnda \
        --site North \
        --telescope LSTN-01 \
        --model_version 7.0.0 \
        --data tests/resources/MLTdata-preproduction.ecsv \
        --parameter_version 1.0.0 \
        --test --psf_hist --cleanup
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: derive_mirror_rnda
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: derive_mirror_rnda_psf_measurement.yml
```

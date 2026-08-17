# simtools-derive-pulse-shape-parameters

```{eval-rst}
.. automodule:: simtools.applications.derive_pulse_shape_parameters
   :members:
   :exclude-members: main
```

```{eval-rst}
Solve (sigma, tau) for a Gaussian convolved with a causal exponential so the
pulse matches user-provided rise and fall widths between fractional amplitude
levels (e.g. 0.1-0.9 rise, 0.9-0.1 fall).

**Command line arguments**

site (str, required)
        North or South.
telescope (str, required)
        Telescope model name.
model_version (str, required)
        Model version.
parameter_version (str, required)
    Parameter version.
rise_width_ns (float, required)
    Rising-edge width between rise_range fractions (ns).
fall_width_ns (float, required)
    Falling-edge width between fall_range fractions (ns).
rise_range (float float, optional)
    Fractional amplitudes (low high) for rise width (default: 0.1 0.9).
fall_range (float float, optional)
    Fractional amplitudes (high low) for fall width (default: 0.9 0.1).
dt_ns (float, optional)
    Time sampling step (ns). Default: 0.1.
time_margin_ns (float, optional)
    Margin added at both ends of readout window. Default: 5.


**Example**

Derive parameters for a pulse with 2.5 ns rise (10-90%) and
 5 ns fall (90-10%) for LSTN-01:

.. code-block:: console

    simtools-derive-pulse-shape-parameters \
    --site North \
    --telescope MSTx-NectarCam \
    --model_version 7.0 \
    --parameter_version 1.0.0 \
    --rise_width_ns 2.5 \
    --fall_width_ns 5.0 \
    --rise_range 0.1 0.9 \
    --fall_range 0.9 0.1 \
    --dt_ns 0.1 \
    --time_margin_ns 10
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: derive_pulse_shape_parameters
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: derive_pulse_shape_parameters.yml
```

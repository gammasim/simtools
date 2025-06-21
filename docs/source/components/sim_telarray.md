# Telescope simulations with sim_telarray

sim_telarray ([MPIK site](https://www.mpi-hd.mpg.de/hfm/~bernlohr/sim_telarray), [CTAO gitlab repository](https://gitlab.cta-observatory.org/Konrad.Bernloehr/sim_telarray)) is used for telescope simulations, including ray tracing, triggering, readout, and camera-level simulation.

The following executables from the `sim_telarray` package are used by `simtools`:

- `sim_telarray`: Main ray-tracing and detector simulation executable.
- `testeff`: Computes telescope and camera optical/electronic efficiency.
- `rx`: Calculates the optical point spread function (PSF) (optional use).
- `corsika_autoinputs`: Generates input parameters for CORSIKA.
- `pfp`: Preprocessor for `sim_telarray`.
- `xyzls`: Simulates a calibration light source at a specified position.
- `ls-beam`: Simulates a laser beam calibration light source.

```{warning}
Incomplete documentation - missing references
```

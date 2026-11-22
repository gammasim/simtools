# Telescope simulations with sim_telarray

sim_telarray ([MPIK site](https://www.mpi-hd.mpg.de/hfm/~bernlohr/sim_telarray), [CTAO gitlab repository](https://gitlab.cta-observatory.org/cta-computing/dpps/simpipe/simulation_software/sim_telarray)) is used for telescope simulations, including ray tracing, triggering, readout, and camera-level simulation.

The following executables from the `sim_telarray` package are used by `simtools`:

- `sim_telarray`: Main ray-tracing and detector simulation executable.
- Camera efficiency is calculated by simtools from model tables; no separate sim_telarray executable
  is used for this calculation.
- `rx`: Calculates the optical point spread function (PSF) (optional use).
- `ff-1m`: Simulates a flat-fielding device.
- `xyzls`: Simulates a calibration light source at a specified position.
- `ls-beam`: Simulates a laser beam calibration light source.

simtools generates sim_telarray configuration files from model parameters and production-grid
settings, then executes sim_telarray as part of production and calibration workflows. The command
interface is intentionally lightweight in this component guide; operational details are documented
in application-level documentation.

Two modes allow to bypass parts of the sim_telarray simulation:

- **bypass_optics 2**: Skips the optical ray-tracing part of the simulation (but includes the treatment of the atmosphere using the direction of the telescope axis pointing)
- **bypass_optics 3**: Skips the optical ray-tracing part of the simulation entirely, including the treatment of the atmosphere.

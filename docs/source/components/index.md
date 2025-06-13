# Components

simtools provides tools to configure, run, and analyze simulations and to manage simulation model parameters. It integrates the components listed below into a complete, modular simulation framework.

## Simulation Software

```{note}
Simulation software is external to simtools and developed by the respective teams. The integration of these components into simtools is done through interfaces, allowing for flexibility and future updates.
```

- [CORSIKA](corsika.md): Air shower simulation software.
- [sim_telarray](sim_telarray.md): Telescope simulation software for ray tracing, triggering, and camera-level simulation.

## Simulation Models

- [simulation models and parameters](simulation_models.md) describe the properties of all relevant elements of the observatory, including site, telescopes, and calibration devices.
- [databases](databases.md): Store simulation model parameters for efficient querying and management.

```{toctree}
:hidden:
:maxdepth: 1
corsika.md
databases.md
sim_telarray.md
simulation_models.md
```

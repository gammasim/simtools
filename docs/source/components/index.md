# Components

simtools provides tools to configure, run, and analyze simulations and to manage simulation model parameters. It integrates the components listed below into a modular simulation framework.

The main components are:

- simulation software
- simulation models and parameters, associated databases, setting and validation procedures

## Simulation Software

The following simulation software packages are used by simtools:

- [CORSIKA](corsika.md): Air shower simulation software.
- [sim_telarray](sim_telarray.md): Telescope simulation software for ray tracing, triggering, and camera-level simulation.
- ROBAST

% TODO add link and description to ROBAST

```{note}
Simulation software is external to simtools and developed by the respective teams. The integration of these components into simtools is done through interfaces, allowing for flexibility and future updates.
```

## Simulation Models

[Simulation models and parameters](simulation_models.md) describe the properties of all relevant elements of the observatory, including site, telescopes, and calibration devices.
[Databases](databases.md) are used to store simulation model parameters for efficient querying and management.

```{toctree}
:hidden:
:maxdepth: 1
corsika.md
sim_telarray.md
simulation_models.md
databases.md
```

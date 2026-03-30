# Simulation Models

```{warning}
Incomplete Documentation.
Note also overlap with model parameters section in [data model](../data-model/index.md).
```

Simulation model parameters describe properties of all relevant elements of the observatory. This includes site,
telescopes, and calibration devices.
The management, definition, derivation, verification, and validation of the simulation model is central to simtools.

Model parameters describe for example:

- Atmospheric model parameters (profile, transmission)
- Geomagnetic field characteristics
- Optical and mechanical telescope properties
- Detector plane settings
- Trigger configurations
- Camera readout parameters
- Simulation production related software parameters

The major components of the simulation model are:

1. **Model parameter schema files:** The schema files define the model parameters and their properties. Every model parameter is defined in a schema file, see the corresponding [data model section](../data-model/model_parameters.md) for details.
2. **Model parameter repository:** The simulation model is stored in a [gitlab repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/simulation-models). Review and revision control of the simulation model is done using this repository.
3. **Model parameter databases:** The simulation models are stored in the simulation models database to ensure query efficiency for simtools applications, see the [databases section](databases.md) for details.
4. **Simulation model setting workflows**: Parameter derivation and validation pipelines implemented in through workflows consisting of simtools applications. Maintained in [simulation model setting repository](https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/simulation-model-parameter-setting).

```{note}
Simulation models are at the core of the CTAO Simulation Pipeline. The responsibility for the values and the correctness of the simulation model parameters is with the CTAO Simulation Team.
```

# Databases

The simtools package uses a prototype MongoDB database for the telescopes and sites model, reference data, derived values and test data.
Access to the DB is handled via a dedicated API module. Access to the DB is restricted, please contact the developers in order to obtain access.

Simulation model parameters are stored in databases, see the description in the [Simulation Model](model_parameters.md#simulation-model) section.

Several different data bases are used:

* model parameters DB (name needs to be indicated by `SIMTOOLS_DB_SIMULATION_MODEL` in your `.envfile` )
* derived values DB includes simtools-derived values

:::{Important}
The structure of the database is currently under revisions and will change in near future.
This documentation is therefore incomplete.
:::

## Browsing the mongoDB database

The mongoDB database can be accessed via the command line interface `mongo` or via a GUI tool like `Robo 3T` or `Studio 3T`.

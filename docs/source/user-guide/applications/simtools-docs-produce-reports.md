# simtools-docs-produce-reports

```{eval-rst}
.. automodule:: simtools.applications.docs_produce_reports
   :members:
   :exclude-members: main
```

```{eval-rst}
Produces markdown reports for the simulation model. The report type is selected with ``--report``:

- ``array_element``: parameters of one telescope (``--site``, ``--telescope``, ``--model_version``),
  or all telescopes, sites or model versions (``--all_telescopes``, ``--all_sites``,
  ``--all_model_versions``); ``--observatory`` adds observatory reports.
- ``calibration``: calibration devices (``--model_version`` or ``--all_model_versions``).
- ``model_parameter``: model parameters per telescope (``--site``, ``--telescope``), or all
  telescopes and sites (``--all_telescopes``, ``--all_sites``).
- ``simulation_configuration``: configuration parameters of a simulation software
  (``--simulation_software``, ``--model_version`` or ``--all_model_versions``).

**Command line arguments**

report (str, required)
    Type of report: array_element, calibration, model_parameter or simulation_configuration.
model_version (str, optional)
    Model version.
all_model_versions (activation mode, optional)
    Produce reports for all model versions.
all_telescopes (activation mode, optional)
    Produce reports for all telescopes.
all_sites (activation mode, optional)
    Produce reports for all sites.
observatory (activation mode, optional)
    Produce array_element reports for an observatory at a given site.
simulation_software (str, optional)
    Simulation software workflow (simulation_configuration reports).

**Example**

Produce the parameter report of one telescope:

.. code-block:: console

    simtools-docs-produce-reports --report array_element --site North --telescope LSTN-01 \\
    --model_version 6.0.2

Produce calibration-device reports for all model versions:

.. code-block:: console

    simtools-docs-produce-reports --report calibration --all_model_versions
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: docs_produce_reports
   :no-heading:
```

## Examples

```{eval-rst}
.. simtools-integration-example::
    :file: docs_produce_reports_array_element_run.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: docs_produce_reports_calibration_run.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: docs_produce_reports_model_parameter_run.yml
```

```{eval-rst}
.. simtools-integration-example::
    :file: docs_produce_reports_simulation_configuration_run.yml
```

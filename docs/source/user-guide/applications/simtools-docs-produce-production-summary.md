# simtools-docs-produce-production-summary

```{eval-rst}
.. automodule:: simtools.applications.docs_produce_production_summary
   :members:
   :exclude-members: main
```

```{eval-rst}
Reads ``info.yml`` files from the simulation-models productions directory
and writes a markdown table of production model versions and their short
descriptions.

**Command line arguments**
simulation_models_path (Path)
    Path to the simulation-models repository root.
output_path (Path)
    Directory for the output file.
output_file (str)
    Output markdown file name.

**Example**
.. code-block:: console

    simtools-docs-produce-production-summary \\
        --simulation_models_path ../simulation-models \\
        --output_path simtools-output/reports/productions \\
        --output_file production_version_descriptions.md
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: docs_produce_production_summary
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: docs_produce_production_summary_run.yml
```

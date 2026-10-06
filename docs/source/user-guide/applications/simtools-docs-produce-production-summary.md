# simtools-docs-produce-production-summary

```{eval-rst}
.. automodule:: simtools.applications.docs_produce_production_summary
   :members:
   :exclude-members: main
```

```{eval-rst}
Reads ``info.yml`` files from the simulation-models productions directory
and writes a markdown table of production model versions and their short
descriptions. Select either a checked-out repository with
``--simulation_models_path`` or a local Git repository with
``--simulation_models_git_path`` and ``--simulation_models_git_revision``.

**Command line arguments**

simulation_models_path (Path)
    Path to a checked-out simulation-models repository. Use this or the Git options.
simulation_models_git_path (Path)
    Path to a local normal, bare, or mirror Git simulation-model repository.
simulation_models_git_revision (str)
    Git tag, ref, or commit to read. Defaults to the dependency catalog revision.
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

    simtools-docs-produce-production-summary \\
        --simulation_models_git_path ../simulation-models.git \\
        --simulation_models_git_revision HEAD \\
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

# simtools-docs-generate-toctree

```{eval-rst}
.. automodule:: simtools.applications.docs_generate_toctree
   :members:
   :exclude-members: main
```

```{eval-rst}
Writes ``index.md`` files with Sphinx toctrees into the documentation source tree given by
``--docs_path``, listing the generated reports of each directory.

**Example**

.. code-block:: console

    simtools-docs-generate-toctree --docs_path docs
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: docs_generate_toctree
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: docs_generate_toctree_run.yml
```

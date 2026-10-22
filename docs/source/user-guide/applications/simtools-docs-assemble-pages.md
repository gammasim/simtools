# simtools-docs-assemble-pages

```{eval-rst}
.. automodule:: simtools.applications.docs_assemble_pages
   :members:
   :exclude-members: main
```

```{eval-rst}
Downloads the documentation builds of all versions listed in a version switcher file
(``--switcher_file``) from ``--package_url`` (URL template with a ``{version}``
placeholder, e.g., the GitLab generic package registry) and assembles them into
``--site_path``. The newest version is also served at the site root; identical files of
different versions are stored once (relative symlinks). The application fails if the
deployed size exceeds ``--size_budget_mb``. With ``--validate_only``, only the version
switcher file is checked.

**Example**

.. code-block:: console

    simtools-docs-assemble-pages \\
        --switcher_file docs/_static/switcher.json \\
        --site_path public \\
        --package_url "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/packages/generic/docs/{version}/docs-site.tar.gz"
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: docs_assemble_pages
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: docs_assemble_pages_validate.yml
```

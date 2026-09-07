# simtools-derive-ctao-array-layouts

```{eval-rst}
.. automodule:: simtools.applications.derive_ctao_array_layouts
   :members:
   :exclude-members: main
```

```{eval-rst}
Retrieves the CTAO array layouts definitions from the CTAO common identifiers repository
and synchronizes them with the model parameter definition in 'array_layouts' for all
CTAO sites.

Requires access to the CTAO common identifiers repository. Future versions of the
common identifiers might be stored in a CTAO technical database.

**Command line arguments**

site (str)
    CTAO site (North or South).
parameter_version (str)
    Model parameter version.
repository_url (str)
    URL or path of the CTAO common identifiers repository.
repository_branch (str)
    Repository branch to use for CTAO common identifiers.
updated_parameter_version (str)
    Updated parameter version.

**Example**

Derive the CTAO array layouts definitions for the North site using the CTAO common identifiers
repository. Merge the retrieved array layouts with the model parameter definition in
'array_layouts' (in this example for site North and parameter version 2.0.0). Write the
updated array layouts and metadata using the parameter version indicated by the
'updated_parameter_version' argument.

.. code-block:: console

    simtools-derive-ctao-array-layouts --site North \
        --repository_url "https://gitlab.cta-observatory.org/" \
        "cta-computing/common/identifiers/-/raw/" \
        --repository_branch main \
        --site North --parameter_version 2.0.0 \
        --updated_parameter_version 3.0.0
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: derive_ctao_array_layouts
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: derive_ctao_array_layouts.yml
```

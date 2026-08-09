# simtools-submit-array-layouts

```{eval-rst}
.. automodule:: simtools.applications.submit_array_layouts
   :members:
   :exclude-members: main
```

```{eval-rst}
Validates that all telescope defined in the array layouts exist in the database for the
specified model version. Prepares both JSON-style model parameters and corresponding
metadata for submission.

**Command line arguments**
array_layouts (str, required)
    Array layouts file.
updated_parameter_version (str, optional)
    Updated parameter version.
input_meta (str, optional)
    Input meta data file(s) associated to input data (wildcards or list of files allowed).
model_version (str, required)
    Model version.

**Example**

Submit and validate a new array layout dictionary:

.. code-block:: console

    simtools-submit-array-layouts \
        --array_layouts array_layouts.json \\
        --model_version 6.0.0 \\
        --updated_parameter_version 0.1.0 \\
        --input_meta array_layouts.metadata.yml
```

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: submit_array_layouts
   :no-heading:
```

## Example

```{eval-rst}
.. simtools-integration-example::
    :file: submit_array_layouts.yml
```

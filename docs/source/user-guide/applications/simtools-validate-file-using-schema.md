# simtools-validate-file-using-schema

```{eval-rst}
.. automodule:: validate_file_using_schema
   :members:
   :exclude-members: main
```

## Overview

This application validates YAML, JSON, and ECSV files against simtools schemas. Use `file_name`
for one file or a filename pattern, or use `file_directory` to validate files in a directory.

When `schema_file` is omitted for model-parameter validation, simtools uses its bundled
model-parameter schemas. The `data_type` option selects the validation mode:

| `data_type` | Purpose |
| --- | --- |
| `data` | Validate ordinary data files. |
| `metadata` | Validate metadata files. |
| `schema` | Validate a schema document against a metaschema. |
| `model_parameter` | Validate model-parameter files using the model-parameter schemas. |

Pass `schema_file` when the schema is not selected automatically. Use `check_exact_data_type` to
require the exact declared data type, and `ignore_software_version` to skip software-version
checks where appropriate.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: validate_file_using_schema
   :no-heading:
```

## Examples

Validate an ECSV data file:

```{eval-rst}
.. simtools-integration-example::
    :file: validate_file_using_schema_ecsv_validate_data.yml
```

Validate a JSON schema document:

```{eval-rst}
.. simtools-integration-example::
    :file: validate_file_using_schema_json_validate_schema-0.2.0.yml
```

Validate a directory of model parameters:

```{eval-rst}
.. simtools-integration-example::
    :file: validate_file_using_schema_validate_directory_of_model_parameters.yml
```

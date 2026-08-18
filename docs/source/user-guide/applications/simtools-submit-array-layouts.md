# simtools-submit-array-layouts

```{eval-rst}
.. automodule:: simtools.applications.submit_array_layouts
   :members:
   :exclude-members: main
```

## Overview

This application validates all telescope elements in an array-layout parameter against the
telescope production table for the selected model version, then writes the validated parameter
and metadata with `updated_parameter_version`.

Use the legacy `array_layouts` option to submit a complete canonical parameter file. Alternatively,
define one new layout directly as a subset of an existing reference layout. Direct submission
reads the requested `parameter_version`, preserves the existing named layouts, and rejects a new
layout name that already exists.

## Direct layout input

The following command adds a South-site layout containing two telescopes from `hyper_array`:

```console
simtools-submit-array-layouts \\
    --site South --model_version 7.0.0 \\
    --parameter_version 3.0.0 \\
    --updated_parameter_version 3.0.99 \\
    --array_layout_name South-dual-camera-example \\
    --array_element_list MSTS-01 MSTS-301
```

`reference_array_layout` defaults to `hyper_array`. The submitted elements must be non-empty,
unique, and contained in that reference layout. Layout names are normalized by replacing spaces
with dashes.

## Legacy file input

Submit a complete array-layout parameter file with:

```console
simtools-submit-array-layouts \\
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

## Integration examples

The legacy file-input workflow is covered by:

```{eval-rst}
.. simtools-integration-example::
    :file: submit_array_layouts.yml
```

The `submit_array_layouts_subset.yml` integration test introduced with the user-defined layout
workflow demonstrates direct subset submission:

```{eval-rst}
.. simtools-integration-example::
    :file: submit_array_layouts_subset.yml
```

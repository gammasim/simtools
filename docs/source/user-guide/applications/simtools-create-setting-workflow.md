# simtools-create-setting-workflow

```{eval-rst}
.. automodule:: simtools.applications.create_setting_workflow
   :members:
   :exclude-members: main
```

Prepare a simple model-parameter setting from a value, version, instrument, and scientific
description. Run from the parameter-setting repository, or select its root with `--output_path`.

```console
simtools-create-setting-workflow \
  --instrument LSTN-design --parameter min_photons \
  --value 121 --parameter_version 2.0.1 \
  --description "Updated photon threshold" --run
```

The command creates `input/INSTRUMENT/PARAMETER/ACTIVITY/config.yml` and `input.meta.yml`.
It generates IDs, timestamps, contact defaults, and an unambiguous site automatically.
Use `--site` for instruments shared between sites and `--source_url` for a reference.
The source reference is included in the scientific description and output provenance.
Select a model repository with the usual `--simulation_models_path` or Git-source options.

Without `--run`, only the inputs are prepared. With it, the existing workflow runner writes
the parameter JSON and metadata under `output/INSTRUMENT/PARAMETER/ACTIVITY/`.
Identical inputs reuse their workflow; conflicting values, descriptions, contacts, or runtime
definitions require a new parameter version. Reruns preserve previous outputs and write under
`output/INSTRUMENT/PARAMETER/ACTIVITY/reruns/EXECUTION_ID/`.

An optional `--runtime_environment_file runtime.yml` copies the validated runtime definition into
the new workflow as `runtime.yml`, without starting a container during preparation. When `--run`
is used, the generated runtime file is selected automatically. Use an image digest for OCI images
and ensure paths in the definition are available from the repository root.
File and structured parameters use the existing derivation/submission applications.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: create_setting_workflow
   :no-heading:
```

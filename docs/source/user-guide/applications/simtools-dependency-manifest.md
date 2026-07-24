# simtools-dependency-manifest

```{eval-rst}
.. automodule:: dependency_manifest
   :members:
   :exclude-members: main
```

## Overview

The application writes the dependency provenance manifest for the active simtools environment.
Production images use it during the image build; users can also run it to capture the installed
Python packages, scientific-software build options, image references, and a canonical SHA-256
digest.

Use `--development` while building a development image that has the source tree but does not
install simtools. In this mode, `--project_file` supplies the direct Python dependencies and
`--build_option_files` supplies available CORSIKA and sim_telarray build metadata.

## Output and schema

| Output | Format | Schema | Purpose |
| --- | --- | --- | --- |
| `--output_file` | JSON | `dependency_manifest.schema` 0.1.0 | Canonical dependency provenance manifest. |
| `--output_file` with `.sha256` appended | Text | None; checksum sidecar | SHA-256 digest of the canonical manifest. |

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: dependency_manifest
   :no-heading:
```

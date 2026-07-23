# simtools-dependency-versions

```{eval-rst}
.. automodule:: simtools.applications.dependency_versions
   :members:
   :exclude-members: main
```

## Overview

The application reads the dependency catalog from `[tool.gammasimtools.dependency-versions]` in
`pyproject.toml`. It validates the catalog and exports the declared Python version, component
releases, optional revisions or digests, archive checksums, and image-build matrices.

Use `--format github-output` in GitHub Actions or local image-build scripts. Use `--format summary`
for a compact view of the values that will be passed to Docker. The catalog represents intended
build inputs; use `simtools-dependency-manifest` to inspect the versions actually present in an
image.

## Output formats and schemas

| Format | Schema | Purpose |
| --- | --- |
| `catalog` | `dependency_versions.schema` 0.1.0 | Validated catalog as formatted JSON. |
| `summary` | None; derived JSON object | Stable scalar values for image builds. |
| `github-output` | None; GitHub Actions key-value assignments | Build scalars and JSON-encoded matrices. |
| `python-requirements` | None; pip requirement lines | Direct Python requirements, optionally including named extras. |

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: dependency_versions
   :no-heading:
```

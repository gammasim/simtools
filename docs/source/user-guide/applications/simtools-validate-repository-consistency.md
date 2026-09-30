# simtools-validate-repository-consistency

```{eval-rst}
.. automodule:: simtools.applications.validate_repository_consistency
   :members:
   :exclude-members: main
```

## Overview

The application validates repository-wide invariants that cannot be checked by validating a
single file against a schema. It checks metadata product references, product identifier content,
workflow version uniqueness, and `SKIP_WORKFLOW_CI` reasons.

Use `--require_git_tracking` in a Git-backed data repository to require referenced products to be
tracked by Git.

## Example

Run from the repository root in CI:

```console
simtools-validate-repository-consistency \
  --repository . \
  --metadata_roots input output \
  --workflow_root input \
  --require_git_tracking
```

The command fails if a metadata file references a missing or untracked product, or if two active
workflows submit the same instrument, parameter, and parameter version.

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: validate_repository_consistency
   :no-heading:
```

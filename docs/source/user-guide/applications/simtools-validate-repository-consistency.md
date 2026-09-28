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

## Command line arguments

```{eval-rst}
.. simtools-cli-help::
   :application: validate_repository_consistency
   :no-heading:
```

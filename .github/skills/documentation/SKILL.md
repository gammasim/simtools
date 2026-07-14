---
name: doc-writing
description: >-
  Write or update simtools documentation using repository conventions,
  including docstrings, Sphinx pages, API reference entries, and changelog
  fragments.
---

# Documentation Writing for simtools

Use this skill when writing or updating documentation in `docs/`, adding or
fixing docstrings in `src/simtools/`, creating user-guide application pages,
or updating API reference pages.

Follow `.github/copilot-instructions.md` for repository-wide rules. This skill
adds a focused documentation workflow and completion checklist.

## When to use this skill

Use this skill for:

1. Writing or updating documentation in `docs/source/`.
2. Improving NumPy-style docstrings in library code under `src/simtools/`.
3. Adding changelog fragments in `docs/changes/` for pull requests.

## Core requirements

1. Public functions, classes, and methods must have docstrings.
2. Use NumPy-style docstrings with relevant sections:
   - `Parameters`
   - `Returns`
   - `Raises`
   - `Examples`
   Include only sections that apply to the documented object.
3. Keep language concise, concrete, and user-focused.
4. Keep line length at 100 characters.
5. Use ASCII text in docs and docstrings.
6. Write documentation pages in MyST Markdown unless an RST file is required
   for application autodoc pages.

## Standard workflow

1. Identify the scope:
   - User-facing behavior change: update user guide page(s).
   - New or changed library module: update API reference.
   - New CLI app: add an application page and toctree entry.
2. Update docstrings in the changed Python files first.
3. Update or add Sphinx documentation pages in `docs/source/`.
4. Add or update the API reference entry for any new module.
5. Add a changelog fragment if the task is part of a PR workflow.
6. Build docs locally and run link checks.

## Required file updates by scenario

### New application in `src/simtools/applications/`

1. Add `docs/source/user-guide/applications/simtools-<app-name>.rst`.
2. Add an entry in `docs/source/user-guide/applications.md` in alphabetical
   order.
3. Ensure the page matches existing application pages: one `.. automodule::`
   directive with `:members:` and usually `:exclude-members: main`.
4. Prefer generated CLI documentation over duplicated command-line help text
   in docstrings.

### New library module in `src/simtools/` (outside `applications/`)

1. Add an API reference entry in the relevant
   `docs/source/api-reference/*.md` file.
2. Use `.. automodule:: <module.path>` with `:members:`.
3. Ensure the new page/section is included by the API reference index where
   needed.

### Pull request changelog

1. Add a short changelog fragment to `docs/changes/<pr-number>.<type>.md`.
2. Allowed types: `feature`, `bugfix`, `api`, `doc`, `maintenance`, `model`.
3. Use PR number (not issue number) in the filename.

## Writing guidance

1. Prefer task-based descriptions: what users do, required inputs, expected
   outputs.
2. Keep examples realistic and aligned with integration test configs under
   `tests/integration_tests/config/` when possible.
3. Do not duplicate implementation details that can drift; link concepts to
   stable CLI options and public APIs.
4. When documenting parameters with units, state expected units clearly and
   consistently.

## Validation checklist

Run after documentation edits:

```bash
cd docs
make html
make linkcheck
```

Run repository checks when docstrings or code-adjacent docs changed:

```bash
pre-commit run --all-files
```

## Done criteria

A documentation task is complete when:

1. All changed public APIs have correct NumPy-style docstrings.
2. User-guide and API reference pages are updated for the change.
3. Required toctree/index entries are present.
4. Changelog fragment is added when applicable.
5. `make html` and `make linkcheck` pass locally.
6. `pre-commit run --all-files` is clean for changed files.

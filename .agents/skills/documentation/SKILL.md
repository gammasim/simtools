---
name: documentation
description: >-
  Write or update simtools documentation using repository conventions,
  including docstrings, Sphinx pages, API reference entries, and changelog
  fragments. Use for changes in docs/source, docs/changes, application
  documentation pages, API reference pages, and NumPy-style docstrings in
  src/simtools.
---

# Documentation Writing for simtools

Follow `AGENTS.md` for repository-wide conventions, environment selection, and linting.
This skill owns documentation-specific instructions. See
`docs/source/developer-guide/documentation.md` for the Sphinx extensions.

## Scope and writing

- Update the user guide for user-facing behavior, the API reference for new or
  moved library modules, and application pages for CLI changes.
- Keep documentation concise and actionable. For small changes, update the relevant
  sentence or example; add a section only for a distinct user task.
- Document public behavior and useful overrides. Keep internal helper names and
  lookup details in code and tests unless users need them to troubleshoot.
- Public functions, classes, and methods need NumPy-style docstrings with only
  relevant `Parameters`, `Returns`, `Raises`, and `Examples` sections. Match the
  signature, return values, and units. Do not add development plans to the docs.
- Write pages in MyST Markdown; use `eval-rst` fences for Sphinx directives.

## Application pages

1. Add `docs/source/user-guide/applications/simtools-<app-name>.md` and an entry
   in `docs/source/user-guide/applications.md` in alphabetical order.
2. Keep the application module docstring to a one-line synopsis. Put inputs,
   outputs, operational details, and examples in the page.
3. Include one autodoc directive, following neighboring pages:

   ````markdown
   ```{eval-rst}
   .. automodule:: simtools.applications.<module_name>
      :members:
      :exclude-members: main
   ```
   ````

4. Generate CLI help with `simtools-cli-help` using `:application: <module_name>`.
   Use `:no-heading:` when the page already supplies the heading; adjust
   `:hide-groups:` or `:show-groups:` only when needed.
5. Render tested examples with `simtools-integration-example` using
   `:file: <config.yml>`. Both custom directives go inside `eval-rst` fences.

## Library API reference

Add new or moved library modules to the relevant `docs/source/api-reference/*.md`
page with `.. automodule:: <module.path>` and `:members:` inside an `eval-rst` fence.
Use the complete import path and match neighboring entries, for example
`model_repository.reader`.

Include every new page in its toctree. A file existing under `docs/source/` is not
sufficient; `toc.not_included` is a documentation failure. Resolve cross-reference
warnings at their source rather than adding broad `nitpick_ignore` or
`suppress_warnings` entries.

## Changelog fragments

For PR work, add a concise fragment to `docs/changes/<pr-number>.<type>.md`.
Use the PR number, not the issue number. Supported types are `feature`, `bugfix`,
`api`, `doc`, `maintenance`, and `model`.

## Validation

For changes to Sphinx pages or docstrings, run from the checkout in the selected
Python environment:

```bash
env PYTHONPATH="$PWD/src" make -C docs clean html linkcheck
```

For Conda, prefix the command with `conda run -n simtools-dev`.
`docs/Makefile` enables `-W -n --keep-going`; treat every Sphinx warning as a failure.
The build imports applications for CLI help and autodoc, so `PYTHONPATH` must point
at this checkout. If a traceback references stale code or `site-packages`, verify
`simtools.__file__` before diagnosing the docs.

Run the repository lint checks specified in `AGENTS.md`.

Before handing off a change that adds, removes, or moves a library module, run
the API coverage check as well. Every reported module must be added to the
appropriate API reference page with an `automodule` entry for its complete
module path, matching the import style already used by that API page:

```bash
python - <<'PY'
from pathlib import Path

src_root = Path("src/simtools")
api_text = "\n".join(
    path.read_text(encoding="utf-8")
    for path in Path("docs/source/api-reference").glob("*.md")
)
missing = []
for path in src_root.rglob("*.py"):
    relative = path.relative_to(src_root)
    if (
        path.name in {"__init__.py", "_version.py"}
        or relative.parts[0] == "applications"
        or relative.parts[0].startswith("_")
    ):
        continue
    module = ".".join(relative.with_suffix("").parts)
    candidates = (module, f"simtools.{module}")
    if not any(f".. automodule:: {candidate}" in api_text for candidate in candidates):
        missing.append(module)
if missing:
    raise SystemExit("Undocumented modules:\n" + "\n".join(sorted(missing)))
PY
```

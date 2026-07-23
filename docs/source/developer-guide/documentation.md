# Generating Documentation

Sphinx is used to create this documentation from the files in the
[docs](https://github.com/gammasim/simtools/tree/main/docs) directory and from the
docstrings in the code.
This is done automatically with each merge into the main branch, see the
[GitHub Action workflow CI-docs](https://github.com/gammasim/simtools/blob/main/.github/workflows/CI-docs.yml).

Docstrings are written in RST format following the [numpydoc](https://numpydoc.readthedocs.io/en/latest/index.html) standard.
Documentation pages are written in Markdown ([MyST flavor](https://myst-parser.readthedocs.io/en/latest/index.html), although RST
is also possible (but not preferred)).

Each application requires a documentation page in
`docs/source/user-guide/applications/`. The page can use either MyST Markdown (`.md`) or RST
(`.rst`).

Keep the application module docstring to a one-line synopsis. This line is used as the command-line
description and included in the application page by `sphinx.autodoc`. Put extended user guidance,
such as input and output descriptions, operational notes, and examples, in the application page.

Application pages can render their CLI reference directly from the parser using the
`simtools-cli-help` directive. By default it hides the repetitive common argparse groups
(`configuration`, `paths`, `execution`, `run time`, `user`) while keeping application-specific
groups visible. Use `:hide-groups:` or `:show-groups:` on the directive to adjust this per page.
Use `:no-heading:` when the surrounding page provides the `Command line arguments` heading.
The directive accepts the short application module name through `:application:` and reads the
module-level `APPLICATION` definition without executing `main()`.

## Building

For writing and testing documentation locally:

```bash
cd docs
make clean
make html
```

This is especially recommended to identify warnings and errors by Sphinx (e.g., from badly formatted
docstrings or RST files). The documentation can be viewed locally in a browser starting from the
file `./build/html/index.html`.

## Hints for Markdown

- links to other files relative to current file: ``[databases](../user-guide/databases.md#databases)`` results in [databases](../user-guide/databases.md#databases).

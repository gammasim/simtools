# Generating Documentation

Sphinx is used to create this documentation from the files in the
[docs](https://github.com/gammasim/simtools/tree/main/docs>) directory and from the
docstrings in the code.
This is done automatically with each merge into the main branch, see the
[GitHub Action workflow CI-docs](https://github.com/gammasim/simtools/blob/main/.github/workflows/CI-docs.yml>).

Docstrings are written in RST format following the [numpydoc](https://numpydoc.readthedocs.io/en/latest/index.html) standard.
Documentation pages are written in Markdown ([MyST flavor](https://myst-parser.readthedocs.io/en/latest/index.html), although RST
is also possible (but not preferred)).

```{caution}
Each application requires a small RST file (e.g., [get_file_from_db.rst](https://github.com/gammasim/simtools/tree/main/docs/source/get_file_from_db.rst]))
to avoid Sphinx warnings regarding duplicated labels (sphinx generates those pages using the sphinx.autodoc extension)
```

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

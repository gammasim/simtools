# Generating Documentation

Sphinx is used to create this documentation from the files in the
[docs](https://github.com/gammasim/simtools/tree/main/docs>) directory and from the
docstrings in the code.
This is done automatically with each merge into the main branch, see the
[GitHub Action workflow CI-docs](https://github.com/gammasim/simtools/blob/main/.github/workflows/CI-docs.yml>).

The documentation is written in Markdown ([MyST flavor](https://myst-parser.readthedocs.io/en/latest/index.html) or reStructuredText (RST) format.
Preference is given to markdown, as more developers are familiar with it.

For writing and testing documentation locally:

```bash
    cd docs
    make clean
    make html
```

This is especially recommended to identify warnings and errors by Sphinx (e.g., from badly formatted
docstrings or RST files). The documentation can be viewed locally in a browser starting from the
file `./build/html/index.html`.

## Hints for RST

Please make sure that you follow the RST format, as sphinx otherwise fails with error messages which are in some cases quite difficult to understand.

Notes

- make sure that headings are underlined with the correct number of `=` characters
- make sure that the indentation is correct and aligned
- use unicode for special characters (e.g., `\u00B2` for superscript 2); see [unicode table](https://unicode-table.com/en/).

## Hints for Markdown

- links to other files and headings should follow this syntax: `` {ref}`model_parameters module <model_parameters>` `` results in {ref}`model_parameters module <model_parameters>`
- links to other files relative to this one in markdown: ``[mongoDB databases](databases.md#databases)`` results in [mongoDB databases](databases.md#databases).

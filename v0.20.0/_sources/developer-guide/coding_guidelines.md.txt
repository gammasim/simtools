# Coding Guidelines

This document describes the coding guidelines for the project.

In general, the simtools project follows the PEP8 style.
It uses the same guidelines outlined in the [ctapipe style guide](https://ctapipe.readthedocs.io/en/latest/developer-guide/style-guide.html)
and [code guidelines](https://ctapipe.readthedocs.io/en/latest/developer-guide/index.html)

## Automatic Formatting

Linting and code checks are done automatically using the pre-commit functionality using `isort`,
`black` and `pyflakes`. As part of the CI workflow Codacy performs a few additional code checks.

It is recommended for developers to install `pre-commit`:

```bash
pre-commit install
```

The configuration of `pre-commit` is defined in
[.pre-commit-config.yaml](https://github.com/gammasim/simtools/blob/main/.pre-commit-config.yaml).

For testing, pre-commit can be applied locally without commit:

```bash
pre-commit run --all-files
```

In rare cases, one might want to skip pre-commit checks with

```bash
git commit --no-verify
```

## Code Linting

Linters of almost all file types are run by the CI-linter workflow.

To run pylint locally, use:

```bash
pylint $(git ls-files 'simtools/*.py')
```

To run ruff locally, use:

```bash
ruff check
```

The options for ruff and pylint are defined in `pyproject.toml`.

### API documentation

Docstrings following the Numpy style must be added to any public function, class or method.
It is also recommended to add docstrings-like comments to private functions.

In applications, the modules should contain docstrings with a general description, command-line
parameters, and examples.
A typical example should look like:

```python
def a_function(parameter):
    """
    Description of what the function is doing.

    Parameters
    ----------
    parameter: type
        description of parameters

    Returns
    -------
    describe return values

    Raises
    ------
    describe exceptions raised

    """

    ...code
```

For reference see the [numpydoc format documentation](https://numpydoc.readthedocs.io/en/latest/format.html).

## Logging

Sufficient logging information should be provided to users and developers. As general guideline, the
following logging levels should be used:

- **INFO**: information useful for the general user about the progress, intermediate results, input or output.
- **WARNING**: information for the general user or developer on something they should know but cannot change.
- **DEBUG**: information only interesting for developers or useful for debugging.
- **ERROR**: something which always leads to an exception or program exit.

Use `logger.error, logger.warning, logger.debug, logger.info`.

## Naming

### Telescope Names

The telescope names as used by simtools follow the pattern "Site-Class-Type", where:

- "Site" is either "North" or "South";
- "Class" is either "LST", "MST", "SCT" or "SST";
- "Type" is a single number ONLY in case of a real telescope existing at the site or a string containing a "D" in case of any other telescope design.

For example:

- "North-LST-1" is the first LST commissioned at the La Palma site, while "North-LST-D234" is the current design of the further 3 LSTs.
- "North-MST-FlashCam-D" and "North-MST-NectarCam-D" are the two MST designs containing different cameras.

Any input telescope names can (and should) be validated by the function validate_array_element_name
(see module {ref}`utils.names <utilsnames>`).
For the Site field, any different capitalization (e.g "south") or site names like "paranal" and
"lapalma" will be accepted
and converted to the standard ones. The same applies to the Class field.
For the Type field, any string will be accepted and a selected list of variations will be converted
to the standard ones.

### Validating names

Names that are recurrently used along the the package should be validated when given as input.
Examples of names are: telescope, site, camera, model version. The functionalities to validate names
are found in  {ref}`utils.names <utilsnames>`. The function \_validate_name receives the input string
and a name dictionary,
that is usually called all_something_names. This dictionary contain the possible names (as keys) and
lists
of allowed alternatives names as values. In case the input name is found in one of the lists, the
key
is returned.

The name dictionaries are also defined in util.names. One should also define specific functions
named
validate_something_names that call the \_validate_name with the proper name dictionary. This is only
meant to
provide a clear interface.

This is an example of a name dictionary:

```bash
all_site_names = {
  "South": ["paranal", "south"],
  "North": ["lapalma", "north"]
}
```

And this is an example of how the site name is validated in the {ref}`telescope_model <telescope-model>` module:

```python
self.site = names.validate_site_name(site)
```

where site was given as parameter to the `TelescopeModel::__init__` function.

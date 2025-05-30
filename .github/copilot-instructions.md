This is a python project in the context of Monte Carlo simulations and high-energy gamma-ray astronomy.

## Guidelines

- use always semantic versions for version strings.
- do not start replies with "this is the final outcome", while it is obviously not.
- do not add trivial comments to the code (e.g. '# Adding five' before calling add_five()). Minimize the number of comments added.
- Use pathlib, not os.path
- Use logging, not print
- Use f-strings for formatting of log messages and other strings.
- Avoid hardcode files.

##  Tests


Unit test files should be in the ./tests/unit_tests/ directory and the directory structure there should follow that of the python package.
For unit tests, do not use test classes but simple test functions.
When writing unit tests, take care of the indentation of the method and the correct formatting.

Check conftest.py for the pytest configuration and common fixtures.

# Test Resources

Test resources are the maintained input and reference files used by unit and
integration tests.
The default development resource set stays in `tests/resources`.

Resource files require updates, e.g., when file formats change or new features are added to the software.

## Types of resource files

There are three types of resource files:

- `tests/resources/static` are static test files maintained by the developers.
- `tests/resources/generated` are generated test files produced by simtools applications.
- `tests/resources/downloaded` are downloaded test files from external sources (e.g., from the simulation models repository).

Whenever possible prefer generated files over manually maintained derived products. Use static files only for inputs that cannot be generated or downloaded.

## Resource files and simtools releases

For a new simtools release, the workflow is:

1. Define a release candidate in `simtools` (e.g., `v0.34.0-rc`). The simtools CI workflow builds and published release-candidate containers for this release candidate.
2. Define a new test resource version in `simtools-tests` (e.g., `v0.34.0`), update the workflow definitions with above container version and generate the new resource set with the release-candidate container.
3. Run the relevant integration tests locally against the new resources.
4. On failure, fix the resource generation workflow or the simtools code and repeat steps 1-3.
5. On success , sync the new resource set into `tests/resources`, and run integration and unit tests again.
6. Release both `simtools` and `simtools-tests`.

This keeps the archived resource set aligned with the released software and
avoids generating release resources from an unpinned development environment.

## Applications

- [`simtools-resources-test-generate`](../user-guide/applications/simtools-resources-test-generate.rst)
  generates a versioned resource set from workflow definitions in
- [`simtools-resources-test-sync`](../user-guide/applications/simtools-resources-test-sync.rst)
  compares a versioned resource set with `tests/resources` and optionally syncs
  it into the repository.

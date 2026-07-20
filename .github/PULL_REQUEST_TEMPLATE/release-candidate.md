# Release Candidate

## Test-file generation

This release might require updates to test files in `tests/resources`.
This is the case when output format or content of the output files generated
by simtools changed.

For updates of test files, the following steps should be performed:

- [ ] define a new version of tests in the [simtools-tests repository](https://github.com/gammasim/simtools-tests).
- [ ] generate new resource files using `simtools-resources-test-generate` and commit them to the simtools-tests repository.
- [ ] run unit and integration tests on the new resources (use `--test-resources-path` / `--test_resources_path` to point pytest at the new bundle)
- [ ] after successful tests, sync the test files into the simtools repository using `simtools-resources-test-sync` and commit the changes to the simtools repository.
- [ ] ensure again that all unit and integration tests pass with the new resources.

## Release Preparation Checklist

- [ ] New test files generated and committed to the simtools-tests repository or confirmed that no new test files are required.
- [ ] All unit and integration tests passed.
- [ ] Release preparation pull request opened from branch `<version string>-rc` (e.g., `v2.1.3-rc`).
- [ ] Pull request confirmed as the final pull request before the release.
- [ ] All notable changes documented in [CHANGELOG.md](https://github.com/gammasim/simtools/blob/main/CHANGELOG.md) using [towncrier](https://towncrier.readthedocs.io/en/stable/):

  ```bash
  towncrier build --yes --version <version string>
  ```

  This updates the changelog using the fragments in the `docs/changes` directory.

- [ ] `CHANGELOG.md` reviewed and confirmed complete.
- [ ] New version added to [docs/_static/switcher.json](https://github.com/gammasim/simtools/blob/main/docs/_static/switcher.json).
- [ ] [CITATION](https://github.com/gammasim/simtools/blob/main/CITATION.cff) reviewed and updated if necessary.

- [ ] Review requested from the simtools team.
- [ ] Pull request approved and ready to merge to `main`.

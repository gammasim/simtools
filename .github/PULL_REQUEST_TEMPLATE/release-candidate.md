# Release Candidate

This pull request should be opened from the branch `<version>-rc`, for example
`v0.36.0-rc`. It should be the final pull request before the release.

## Test-resource generation

This release might require updates to test resources in the `simtools-tests` repository when
simtools changes the output format or content of generated files.

If test resources must be updated:

- [ ] Define a new test-resource version in the [simtools-tests repository](https://github.com/gammasim/simtools-tests).
- [ ] Update its workflow configuration to use the release-candidate `simtools-prod` image, including the required image variant and tag.
- [ ] Generate the new resources with `simtools-resources-test-generate` and commit them to `simtools-tests`.
- [ ] Run unit and integration tests with the new resources using `--test_resources_path`.

## Release Preparation Checklist

- [ ] New test resources are committed to `simtools-tests`, or no resource update is required.
- [ ] All unit and integration tests pass.
- [ ] The release-candidate pull request is confirmed as the final pull request before the release.
- [ ] All notable changes documented in [CHANGELOG.md](https://github.com/gammasim/simtools/blob/main/CHANGELOG.md) using [towncrier](https://towncrier.readthedocs.io/en/stable/):

  ```bash
  towncrier build --yes --version v0.36.0
  ```

  Replace `v0.36.0` with the final release version, not the release-candidate version.
  This updates the changelog using the fragments in the `docs/changes` directory.

- [ ] `CHANGELOG.md` reviewed and confirmed complete.
- [ ] New version added to [docs/_static/switcher.json](https://github.com/gammasim/simtools/blob/main/docs/_static/switcher.json).
- [ ] [CITATION](https://github.com/gammasim/simtools/blob/main/CITATION.cff) reviewed and updated if necessary.

- [ ] Review requested from the simtools team.
- [ ] Pull request approved and ready to merge to `main`.

## After merging the release-candidate pull request

- [ ] Create and push a release-candidate tag, for example `v0.36.0-rc1`.
- [ ] Create the corresponding GitHub release and mark it as a pre-release. The `build-simtools-prod` workflow is
      configured for both candidate-tag pushes and published releases; verify the expected image before generating
      test resources.

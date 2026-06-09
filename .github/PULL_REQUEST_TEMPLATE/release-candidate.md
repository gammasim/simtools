# Release Candidate

## Release Preparation Checklist

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

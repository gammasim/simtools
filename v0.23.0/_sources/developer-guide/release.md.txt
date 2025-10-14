# Prepare and Deploy

```{note}
The steps described in this document are intended for maintainers of the simtools project.
```

Simtools releases follow the [Semantic Versioning 2.0.0](https://semver.org/) guidelines with a version string format of `vMAJOR.MINOR.PATCH` (e.g., `v2.1.3`).
Note that `simtools` is named `gammasimtools` on PyPi and in the conda.

## Release Preparation

1. Consider a release only after all tests are passing.
2. Open a pull request to prepare the release with the branch name `<version string>-rc` (e.g., `v2.1.3-rc`). This should be the final pull request before a release.
3. All notable changes to the simtools project must be documented in the [CHANGELOG.md](https://github.com/gammasim/simtools/blob/main/CHANGELOG.md) file. To update the changelog, run the changelog workflow using [towncrier](https://towncrier.readthedocs.io/en/stable/):

   ```bash
   towncrier build --yes --version <version string>
    ```

    This updates the changelog using the fragments in the docs/changes directory.

4. Review the `CHANGELOG.md` file and ensure that all changes are documented.
5. Add a new line to [docs/_static/switcher.json](docs/_static/switcher.json) indicating the new version.
6. Review the [CITATION](https://github.com/gammasim/simtools/blob/main/CITATION.cff) and apply any changes if necessary.
7. Request a review of the pull request from the simtools team. Merge to main after approval.

## Release and Deploy

1. Prepare a GitHub release with the version number and a summary of the changes ([simtools release pages](https://github.com/gammasim/simtools/releases)) and release.
2. Pypi deployment is triggered automatically by the CI/CD pipeline via the [pypi.yml](https://github.com/gammasim/simtools/blob/main/.github/workflows/pypi.yml) workflow.
3. Container images are built and tagged automatically with the version number, and pushed to [gammasim/simtools](https://github.com/orgs/gammasim/packages?repo_name=simtools) via the [build-simtools-production-images.yml](https://github.com/gammasim/simtools/blob/main/.github/workflows/build-simtools-production-image.yml) workflow.
4. A DOI is issued automatically by Zenodo, see the [simtools Zenodo page](https://zenodo.org/records/15630484).

## Conda feedstock

The conda feedstock for simtools is maintained in [this repository](https://github.com/conda-forge/gammasimtools-feedstock).
A new pull request is automatically created for new releases on PyPi.

New, updated, or removed command-line tools require manual modifications in the `recipe/meta.yaml` file.
Changed dependencies require manual modifications in a similar way.
A template for the required changes can be obtained using the [grayskull](https://pypi.org/project/grayskull/) tool: `grayskull pypi gammasimtools` generates a `meta.yaml` file with the required changes.

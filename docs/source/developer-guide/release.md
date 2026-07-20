# Prepare and Deploy

```{note}
The steps described in this document are intended for maintainers of the simtools project.
```

Simtools releases follow the [Semantic Versioning 2.0.0](https://semver.org/) guidelines with a version string format of `vMAJOR.MINOR.PATCH` (e.g., `v2.1.3`).
Note that `simtools` is named `gammasimtools` on PyPi and in the conda.

## Release Preparation

Open a release candidate pull request from a branch named `<version string>-rc` (e.g., `v2.1.3-rc`).
To use the release-candidate template, append the query parameters
`?quick_pull=1&template=release-candidate.md` to the compare URL. For example:

```text
https://github.com/gammasim/simtools/compare/main...v2.1.3-rc?quick_pull=1&template=release-candidate.md
```

The template is the [Release Candidate pull request
template](https://github.com/gammasim/simtools/blob/main/.github/PULL_REQUEST_TEMPLATE/release-candidate.md), which includes a checklist of items to complete before the release.
This should be the final pull request before a release.

## Release and Deploy

1. Prepare a GitHub release with the version number and a summary of the changes ([simtools release pages](https://github.com/gammasim/simtools/releases)) and release.
2. Pypi deployment is triggered automatically by the CI/CD pipeline via the [pypi.yml](https://github.com/gammasim/simtools/blob/main/.github/workflows/pypi.yml) workflow.
3. Container images are built and tagged automatically with the version number, and pushed to [gammasim/simtools](https://github.com/orgs/gammasim/packages?repo_name=simtools) via the [build-simtools-production-images.yml](https://github.com/gammasim/simtools/blob/main/.github/workflows/build-simtools-production-image.yml) workflow.
4. A DOI is issued automatically by Zenodo, see the [simtools Zenodo page](https://zenodo.org/records/15630484).

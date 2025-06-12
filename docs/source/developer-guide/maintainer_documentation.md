(releases)=

# Maintainer documentation

## Changelog

All notable changes to the simtools project are documented in the CHANGELOG file.

Before a release, the maintainer is building the changelog using the following steps:

```console
towncrier build --yes --version <version string>
```

The maintainer should review the `CHANGELOG.md`, apply any necessary changes, and open a release pull request. This should be the last pull request before a release.
The change files in the `docs/changes` directory will be deleted with the release pull request.

## Releases

Simtools releases are versioned following the [Semantic Versioning 2.0.0](https://semver.org/) guidelines.

To prepare a release, the following steps are required:

1. Open a pull request to prepare a release.  Run the changelog workflow using Towncrier to add entries to `CHANGELOG.md`. This should be the last pull request to be merged before making the actual release.
2. Add a new line to [docs/_static/switcher.json](docs/_static/switcher.json) indicating the new version.
3. Prepare a GitHub release with the version number and a summary of the changes.
4. Pypi deployment is done automatically by the CI/CD pipeline.
5. Docker images are automatically built, tagged with the version number, and pushed to the [gammasim/simtools](https://github.com/orgs/gammasim/packages?repo_name=simtools).
6. A DOI is issued automatically by Zenodo.

## Conda feedstock

The conda feedstock for gammasimtools is [this repository](https://github.com/conda-forge/gammasimtools-feedstock).
A new merge request is automatically created for new releases on pypi.

New, updated, or removed command-line tools require manual modifications in the `recipe/meta.yaml` file.
Changed dependencies require manual modifications in a similar way.
A template for the required changes can be obtained using the [grayskull](https://pypi.org/project/grayskull/) tool: `grayskull pypi gammasimtools` generates a `meta.yaml` file with the required changes.

(releases)=

# Maintainer documentation

## Releases

Simtools releases are versioned following the [Semantic Versioning 2.0.0](https://semver.org/) guidelines.

To prepare a release, the following steps are required:

1. Open a pull request to prepare a release. This should be the last pull request to be merged before making the actual release.
1. Prepare a github release with the version number and a summary of the changes.
1. Pypi deployment is done automatically by the CI/CD pipeline.
1. Docker images are automatically built, tagged with the version number, and pushed to the [gammasim/simtools](https://github.com/orgs/gammasim/packages?repo_name=simtools).
1. A DOI is issued automatically by Zenodo.

# Dependency versions and provenance

simtools separates declared dependency versions from the versions observed in a built image.
This allows normal Python installations to retain compatible dependency ranges while production
containers use declared software releases and simulation products remain traceable.

## Sources of truth

The dependency information is maintained in two places with distinct responsibilities:

- `[project.dependencies]` and `[project.optional-dependencies]` in `pyproject.toml` declare the
  supported direct Python requirements.
- `dependency_versions.yml` declares the supported Python version, container base images, scientific
  software releases, archive checksums, the default simulation-model version, and the
  `simtools-tests` repository URL, Git ref, and resource-directory version.

Dockerfiles do not provide independent version defaults. GitHub Actions reads the catalog with

```console
simtools-dependency-versions --format github-output
```

and supplies the resulting image references and build arguments.

## Updating versions

Change the compatible Python requirements in `pyproject.toml` or the external component entry in
`dependency_versions.yml`.
For every Git source, specify a readable `ref` (or component-specific `source-ref`).
Use release tags for released software; branches can be used during development.
Build workflows clone the selected ref and record the actual commit in build provenance.
There is no lock file and no need to look up or maintain Git SHAs.
Rebuilding a release follows the tag's current target, so tags should be kept stable upstream.
`simtools-tests` also has `resource-version`: this is the release-named directory selected
inside its checkout, not a Git ref. CORSIKA branch refs need an explicit numeric `build-id`, because
image names use that identifier. sim_telarray refs that are not already valid OCI tags likewise
need a safe `build-id`; this keeps the source ref readable without putting `/` into image or
artifact names. Archive SHA-256 checksums validate downloaded files. Optional OCI image digests
and Git revisions remain supported for installations that need explicit pins.
An optional interaction-table `revision` is used by integration CI, with the readable `ref`
remaining in the catalog. The simulation-model and test-resource branches follow the CI selections
described below.

Production image tags longer than 128 characters are shortened automatically. The tag retains
its first 111 characters followed by a 16-character digest of the full tag, keeping long dependency
identifiers distinct. The exported tag used for image tests is the same tag published by the build.

For a release, update the tags and resource version in the catalog, run CI, and tag simtools.
The catalog is packaged with simtools; built images retain the observed commits in their manifest.

Install the compatible Python environment in a clean Python 3.14 environment containing all extras:

```console
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,doc,tests]'
```

The resulting Python and package versions are recorded in the image dependency manifest. Image
builds use catalogued Git refs and retain both the refs and observed commits in build metadata.

Validate the catalog and matrices with

```console
simtools-dependency-versions --format summary
simtools-dependency-versions --format catalog
```

## Container manifest

Production and development images contain the canonical dependency record at
`/opt/simtools/provenance/dependency-manifest.json`. It contains the simtools revision, Python and
direct Python dependency versions, scientific build options, observed source
revisions, and parent-image references. Credentials, local paths, hostnames, and build
timestamps are excluded.

Inspect the active environment with any simtools application:

```console
simtools-simulate-prod --build_info
```

Applications supporting the common output argument can export the complete record with

```console
simtools-simulate-prod --export_build_info build-info.yml [OTHER OPTIONS]
```

For an Apptainer container pulled from the OCI image, the same information is available with:

```console
apptainer exec simtools-prod.sif \
  python -c 'from simtools.dependencies import get_dependency_manifest_digest; print(get_dependency_manifest_digest())'
apptainer inspect --json --labels simtools-prod.sif
```

Published production images include `/opt/simtools/provenance/dependency-manifest.json.sha256`, containing the dependency manifest digest.

## Runtime configuration

`.env_template` supplies runtime defaults and example paths; `.env` remains local and ignored.
The dependency catalog supplies the model repository URL and default ref, as well as the
default `simtools-tests` version. `.env` supplies local model-repository paths and user settings;
the catalog remains the fallback for the model ref.

## CI repository branches

Test CI selects the `simulation-models` and `simtools-tests` branches independently of the
catalog's release refs. Both default to `main`, including release and release-candidate runs.
Manual unit, integration, benchmark, and production-image test workflows accept
`simulation_model_branch` and `simtools_tests_branch`. The reusable integration workflow accepts
the same inputs, and its callers forward both selections.

For pull-request, scheduled, and release runs, repository variables `SIMULATION_MODEL_BRANCH`
and `SIMTOOLS_TESTS_BRANCH` can override `main`. Explicit workflow inputs take precedence;
manual and reusable inputs default to `main`. The shared test setup action accepts
`simulation-model-branch` and `simtools-tests-branch` with the same default.

The test-resource directory remains selected by the catalog's `resource-version`;
it is independent of the branch checked out. Custom test branches must contain that directory.
Unit tests and unit benchmarks export these settings; integration tests download the selected
repositories. Image builds use the catalog's source refs.


## Runtime catalog selection

Runtime applications prefer the dependency catalog bundled with the running simtools source or
installation over a catalog in the working directory. This keeps container defaults tied to the
software installed in the image when a host checkout is mounted. Working-directory lookup is a
fallback when no bundled catalog exists.

Set `SIMTOOLS_DEPENDENCY_VERSIONS` to explicitly select another catalog, including a mounted host
catalog. A missing override file raises an error. An explicit `start_path` passed to
`find_dependency_versions` searches that directory and its parents before bundled catalogs.

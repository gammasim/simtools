# Dependency versions and provenance

simtools separates declared dependency versions from the versions observed in a built image.
This allows normal Python installations to retain compatible dependency ranges while production
containers remain repeatable and simulation products remain traceable.

## Sources of truth

The dependency information is maintained in two places with distinct responsibilities:

- `[project.dependencies]` and `[project.optional-dependencies]` in `pyproject.toml` declare the
  supported direct Python requirements.
- `dependency_versions.yml` declares the supported Python version, container base images, scientific
  software releases, archive checksums, the default simulation-model version, and the
  `simtools-tests` repository URL, Git ref and revision, and resource-directory version.

Dockerfiles do not provide independent version defaults. GitHub Actions reads the catalog with

```console
simtools-dependency-versions --format github-output
```

and supplies the resulting image references and build arguments.

## Updating versions

Change the compatible Python requirements in `pyproject.toml` or the external component entry in
`dependency_versions.yml`.
For every Git source, keep both a readable `ref` (or component-specific `source-ref`) and a
40-character `revision`. The ref may be a release tag or branch name and is used for review and
auditing. Image workflows check out the revision, so a moved tag or branch cannot silently change
a build. `simtools-tests` also has `resource-version`: this is the release-named directory selected
inside its checkout, not a Git ref. CORSIKA branch refs also need an explicit numeric `build-id`,
because image names use that identifier. OCI image digests and archive SHA-256 values serve the same
immutable role for their respective inputs.

When updating a Git dependency, resolve the intended ref first and record its commit:

```console
git ls-remote https://example.org/group/project.git refs/tags/v1.2.3
```

Use the full returned SHA as `revision`. The catalog validator checks its shape; CI verifies that
the fetched checkout is exactly that commit.

Install the compatible Python environment in a clean Python 3.14 environment containing all extras:

```console
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,doc,tests]'
```

The resulting Python and package versions are recorded in the image dependency manifest. Image
builds use catalogued Git revisions and retain the readable refs in build metadata.

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
The dependency catalog supplies the model repository URL and pinned revision, as well as the
default `simtools-tests` version. `.env` supplies local model-repository paths and user settings;
the catalog remains the fallback for the pinned revision.

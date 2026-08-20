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
  `simtools-tests` repository URL and version.

Dockerfiles do not provide independent version defaults. GitHub Actions reads the catalog with

```console
simtools-dependency-versions --format github-output
```

and supplies the resulting image references and build arguments.

## Updating versions

Change the compatible Python requirements in `pyproject.toml` or the external component entry in
`dependency_versions.yml`.
For external sources, update the human-readable release. An optional Git revision, OCI image
digest, or archive SHA-256 can be added when an immutable build input is required. Dockerfiles
always record the archive checksum and source revisions actually used for a build.

Install the compatible Python environment in a clean Python 3.14 environment containing all extras:

```console
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,doc,tests]'
```

The resulting Python and package versions are recorded in the image dependency manifest. Image
builds use the catalogued release tags unless an optional revision or digest is declared.

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
The dependency catalog supplies the default model database name and version, as well as the
default `simtools-tests` version. `.env` may override the model database name or version and
supplies local paths, credentials, and user settings. The catalog remains the fallback when no
override is present.
Database credentials, server addresses, and user information are never copied into an image or
dependency manifest.

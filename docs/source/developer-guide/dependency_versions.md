# Dependency versions and provenance

simtools separates declared dependency versions from the versions observed in a built image.
This allows normal Python installations to retain compatible dependency ranges while production
containers remain repeatable and simulation products remain traceable.

## Sources of truth

The dependency information is maintained in two places with distinct responsibilities:

- `[project.dependencies]` and `[project.optional-dependencies]` in `pyproject.toml` declare the
  supported direct Python requirements.
- `[tool.gammasimtools.dependency-versions]` declares the supported Python version, container base images,
  scientific software releases and revisions, archive checksums, image digests, and the default
  simulation-model version.

Dockerfiles do not provide independent version defaults. GitHub Actions reads the catalog with

```console
simtools-dependency-versions --format github-output
```

and supplies the resulting immutable image references and build arguments.

## Updating versions

Change the compatible Python requirements or the external component entry in `pyproject.toml`.
For external sources, update both the human-readable release and the immutable Git revision,
image digest, or archive SHA-256. Never use `main`, `master`, or `latest` as a production input.

Install the compatible Python environment in a clean Python 3.14 environment containing all extras:

```console
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,doc,tests]'
```

The resulting Python and package versions are recorded in the image dependency manifest. Image
builds still consume only immutable external sources and image digests from the catalog.

Validate the catalog and matrices with

```console
simtools-dependency-versions --format summary
simtools-dependency-versions --format catalog
```

## Container manifest

Production and development images contain the canonical dependency record at
`/opt/simtools/provenance/dependency-manifest.json`. It contains the simtools revision, Python and
direct Python dependency versions, scientific build options, source
revisions, and immutable parent-image references. Credentials, local paths, hostnames, and build
timestamps are excluded.

Inspect the active environment with any simtools application:

```console
simtools-simulate-prod --build_info
```

Applications supporting the common output argument can export the complete record with

```console
simtools-simulate-prod --export_build_info build-info.yml [OTHER OPTIONS]
```

For Apptainer, the same information is available without extracting the SIF file:

```console
apptainer exec simtools-prod.sif \
  python -c 'from simtools.dependencies import get_dependency_manifest_digest; print(get_dependency_manifest_digest())'
apptainer inspect --json --labels simtools-prod.sif
```

The production workflow verifies the manifest after OCI-to-SIF conversion and publishes a
provenance artifact mapping the release, Git revision, OCI digest, SIF digest, and
dependency-manifest digest. Published OCI indexes also carry the manifest digest as the
`org.gammasim.simtools.dependency-manifest-sha256` annotation.

## Runtime configuration

`.env_template` supplies runtime defaults and example paths; `.env` remains local and ignored.
Only the selected simulation-model database name and resolved model version enter runtime
provenance. Database credentials, server addresses, and user information are never copied into an
image or dependency manifest.

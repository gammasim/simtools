# Dependency versions and provenance

`pyproject.toml` declares Python dependencies. `dependency_versions.yml` declares the supported
Python version, container base image, external software refs, model compatibility, and build
inputs. Workflows use this catalog to build images and record the resolved commits in the image
manifest.

## Update dependencies

Change Python requirements in `pyproject.toml` and external software versions in
`dependency_versions.yml`. Use stable tags for releases; use branches for development. Keep refs
stable upstream so rebuilding a release resolves to the same source. The catalog supports
optional Git revisions and OCI digests for explicit pins, and requires SHA-256 checksums for
downloaded archives.

`simtools-tests.resource-version` selects a resource directory inside the test repository; it is
separate from the repository Git ref. CORSIKA and sim_telarray refs that cannot be used in image
tags need a `build-id`.

Install the supported environment and validate catalog values with:

```console
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,doc,tests]'
simtools-dependency-versions --format summary
```

## Image provenance

Production and development images store their dependency manifest at
`/opt/simtools/provenance/dependency-manifest.json`. It records the simtools revision, installed
Python dependencies, build options, source refs and resolved commits, and parent images. It omits
credentials, local paths, hostnames, and timestamps.

Inspect the running environment with:

```console
simtools-simulate-prod --build_info
simtools-simulate-prod --export_build_info build-info.yml
```

For an Apptainer image, inspect its labels with `apptainer inspect --json --labels simtools-prod.sif`.

## Runtime catalog

The installed simtools catalog supplies the model repository URL and default ref. `.env_template`
shows runtime defaults; local paths and settings belong in the ignored `.env` file.

Runtime applications prefer the catalog bundled with simtools. Set
`SIMTOOLS_DEPENDENCY_VERSIONS` to select another catalog. A missing override file is an error.
When explicitly searching from a directory, `find_dependency_versions` checks that directory and
its parents before bundled catalogs.

## CI test refs

CI uses `model-repository.default-ref` from `dependency_versions.yml` unless overridden by the
`SIMULATION_MODEL_BRANCH` repository variable or a workflow input. The `simtools-tests` Git ref
defaults to `main` and can be overridden with `SIMTOOLS_TESTS_BRANCH` or its workflow input. Its
resource directory is selected independently by `resource-version`.

## Model compatibility

Starting with catalog schema `0.8.0`, `model-repository.compatible-versions` defines accepted
stable model releases. `default-ref` is the recommended release and must satisfy this range:

```yaml
model-repository:
  name: CTAO-Simulation-Model
  default-ref: v0.18.0
  compatible-versions: "==0.18.0"
```

Use packaging specifiers such as `>=0.18.0,<0.19.0`; expand the range only after confirming the
admitted releases are compatible. Catalogs without a range accept only their declared default.
This lets a newer submission controller validate against an older image's bundled catalog.

At runtime, simtools checks that the selected Git checkout has exactly one stable release tag and
that its version is allowed by the running installation's catalog. Use
`--simulation_models_path /path/to/simulation-models` to select a checkout. The check does not
switch refs or validate local edits and model contents. A directory without Git release metadata
cannot be checked. `--test` permits development inputs with a warning; they are not qualified for
production.

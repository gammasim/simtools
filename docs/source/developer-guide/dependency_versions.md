# Dependency versions and provenance

`pyproject.toml` declares Python dependencies. `dependency_versions.yml` declares the supported
Python version, the container base image, the simulation model and test references, and the
CORSIKA and sim_telarray images that simtools consumes. Workflows use this catalog to select
images and to record the dependency manifest of the images they build.

simtools does not build CORSIKA or sim_telarray. Their images are built in the
simulation-containers repository, which pins the sources, configuration, and patches used for
each image. The catalog only names the published images.

## What the catalog declares

| Section | Declares | Used by |
| --- | --- | --- |
| `images` | Registry and tag suffix of the CORSIKA and sim_telarray images | Production and development image builds |
| `corsika`, `sim-telarray` | Component version (`source-ref`), optional `build-id`, optional image digests | Production matrix, development images |
| `production-combinations` | CORSIKA and sim_telarray versions per production image, optional CPU variants | Production matrix |
| `base-image`, `python`, `apptainer` | Container base image and tool versions | Image builds, summary export |
| `corsika-interaction-tables` | Interaction tables reference | Integration test setup |
| `model-repository` | Simulation model repository, default ref, compatible release range | CI and runtime model checks |
| `simtools-tests` | Test repository, ref, and resource directory version | CI and runtime test resources |
| `cpu-variants` | CPU variants of the images | Production matrix |

## Update dependencies

Change Python requirements in `pyproject.toml`. Change images in `dependency_versions.yml`:

- `images.registry` and `images.tag-suffix` set where the images are published and the suffix
  added to their tags.
- `corsika` and `sim-telarray` list one entry per component version. The `source-ref` is the
  version identity used by `production-combinations`.
- `production-combinations` pairs one CORSIKA and one sim_telarray version per production image.

For example, this catalog fragment:

```yaml
images:
  registry: ghcr.io/gammasim
  tag-suffix: -gl1
corsika:
- source-ref: v7.8010
sim-telarray:
- source-ref: v2025-11-30-rc
```

resolves to these image references for the `avx2` CPU variant:

```text
ghcr.io/gammasim/corsika7:v78010-avx2-gl1
ghcr.io/gammasim/sim_telarray:v2025-11-30-rc-gl1
```

Image tags use the build ID of each component. A CORSIKA `source-ref` of the form
`v<major>.<minor>` gives the build ID by removing the `v` and the dot (`v7.8010` gives `78010`).
A sim_telarray `source-ref` that is not a valid image tag, for example one containing a slash,
requires an explicit `build-id`. Build IDs in the production matrix do not include the tag suffix.

An image digest pins an image instead of its tag. `image-digests` (per CPU variant) applies to
CORSIKA and `image-digest` applies to sim_telarray. A declared digest replaces the tag in the
image reference, and the tag suffix does not apply to digest references.

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

`model-repository.compatible-versions` defines the accepted stable model releases.
`default-ref` is the recommended release and must satisfy this range:

```yaml
model-repository:
  name: CTAO-Simulation-Model
  default-ref: v0.18.0
  compatible-versions: "==0.18.0"
```

Use packaging specifiers such as `>=0.18.0,<0.19.0`; expand the range only after confirming the
admitted releases are compatible. This lets a newer submission controller validate against an
older image's bundled catalog.

At runtime, simtools checks that the selected Git checkout has exactly one stable release tag and
that its version is allowed by the running installation's catalog. Use
`--simulation_models_path /path/to/simulation-models` to select a checkout. The check does not
switch refs or validate local edits and model contents. A directory without Git release metadata
cannot be checked. `--test` permits development inputs with a warning; they are not qualified for
production.

## Catalog schema

`schema_version` selects the schema document in `src/simtools/schemas/dependency_versions.schema.yml`.
Each schema version is a separate YAML document in that file, and the validator accepts every
listed version.

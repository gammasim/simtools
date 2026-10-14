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
catalog's release refs. Development runs default to `main`. Published release runs select the model catalog's
`default-ref`; the test-resource branch remains independently selected.
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

## Supported model releases

The catalog bundled with each simtools installation defines its model compatibility
contract. Starting with catalog schema `0.8.0`, `model-repository` requires
`compatible-versions`, a `packaging.specifiers.SpecifierSet` expression, for example:

```yaml
model-repository:
  name: CTAO-Simulation-Model
  default-ref: v0.18.0
  compatible-versions: "==0.18.0"
```

The default selects a recommended input; the range restricts supported releases;
production provenance identifies the exact release, commit, and content SHA-256
actually used. A leading `v` is normalized. Stable release defaults must satisfy
the constraint. Prereleases and development refs are not qualified production
inputs, even when a broad numeric constraint would otherwise include them.
Python package requirements continue to live in `pyproject.toml`. Existing
scientific build combinations remain explicit; a model range does not widen them.

For historical catalogs without a range, the declared release default is the only
accepted production release. Legacy `model-database.default-tag` entries are
supported. Thus the unchanged simtools v0.37.0 image accepts models v0.17.1 and
rejects v0.17.2 before production submission. This conservative fallback is a
policy, not evidence that every other release fails intrinsically.

A filesystem checkout must have one unambiguous stable release version tagged at
HEAD and unchanged model contents. Exported directories without Git metadata need
`simulation-models-release.json` at their root, with `version`, the 40-character
source `commit`, and `content_sha256`. The digest hashes, in sorted path order,
each model file's repository-relative UTF-8 name, a NUL byte, and the raw SHA-256
of its contents. Only `simulation-models/` is included. Symbolic links are rejected.
The manifest is verified, not trusted solely on its version label. Changing files
without producing a new identified snapshot is not a qualified production input.

For HTCondor production grids, the submission controller reads the selected image's
bundled catalog using Apptainer and reports its simtools version separately from
the submit host. Before simulation jobs are submitted, it initializes representative
configurations through the selected backend and image, including each selected
zenith/atmosphere mode. This loads models and constructs the simulator without
running showers. Initialization outputs go into a sibling `work/<output-name>-dependency-validation/`
directory, outside the production tree so collection cannot mistake canary submissions
for production jobs. A filesystem snapshot's identity is checked again
on workers using standard-library code that also runs in historical images.
No image is patched or rebuilt, and no model checkout is switched automatically.

Normal logs show the selected version/path, constraint and policy, target runtime,
and initialization status. `dependency-compatibility.json` records the decision
and exact identities; science-test reports include the same compatibility result.
Dry runs show the static decision and label initialization as not checked, without
submitting canaries. Initialization is mandatory even without the optional host
filesystem `--preflight` checks. Validation is deliberately rerun for each submission;
there is no implicit cache of previous successful campaigns.

Direct simulations validate the selected models against the running installation's
catalog before model reading. Existing `--test` mode permits development inputs
with a conspicuous unqualified status; it must not be used for qualified production.
Git source reads retain their fixed source identity; filesystem production commands
receive only `--simulation_models_path`, never Git source options.

### Model format evolution

Model repository release versions and scientific model versions (such as `7.0.0`)
identify different things. Neither replaces a reader-format compatibility contract.
Future simulation-models release metadata should declare an independent format
version, and readers should declare which formats they support. Layout changes
must be treated as breaking even when parameter values are unchanged. This producer
policy requires agreement in the simulation-models repository; this change does not
edit existing tags or claim that such metadata already exists. Until that policy
is adopted, explicit release constraints and initialization checks remain the gate.

Release CI must test catalog-default model refs. Development branch testing remains
separate and cannot silently widen a published compatibility contract. Expand a
range only with recorded compatibility evidence, including the releases it admits
and the next excluded release. Existing production image manifests describe built-in
dependencies; mounted model inputs are recorded separately in campaign provenance.

# Build Images

Pre-built OCI and Apptainer images are available from the
[simtools package registry](https://github.com/orgs/gammasim/packages?repo_name=simtools).
The GitHub Actions workflows in `.github/workflows/build-*.yml` are the reference image builds.

All build versions come from `pyproject.toml`; see
[Dependency versions and provenance](dependency_versions.md). Dockerfiles deliberately have no
independent software-version defaults. Before a local build, export the validated values with

```console
simtools-dependency-versions --format github-output
```

## Scientific component images

`docker/Dockerfile-corsika7` builds each catalogued CORSIKA and CPU variant. It requires the CORSIKA
source token and the `autoconf.tar.gz` archive. The Dockerfile records the source,
configuration and optimization-patch revisions that it actually checked out, as well as the
archive checksum it calculated. Optional catalogued revisions and checksums are verified when set.

`docker/Dockerfile-simtel_array` builds the catalogued sim_telarray, hessio and stdtools releases.
It requires the corresponding GitLab tokens and `gsl.tar.gz`; its calculated archive checksum is
recorded before extraction, and verified when the catalog supplies one.

Use the workflow-generated matrix values as build arguments. This ensures that a local build uses
the same base-image tags, source releases and flags as CI. Add optional digests and revisions to
the catalog when a fully immutable build is needed.

## Production and development images

`docker/Dockerfile-simtools-prod` installs the checked-out simtools revision with the compatible
Python dependencies declared in `pyproject.toml`. Its CORSIKA, sim_telarray and AlmaLinux inputs
are version-tag references unless optional OCI digests are declared.

`docker/Dockerfile-simtools-dev` installs the same compatible Python dependencies, including the
development, documentation and test extras, but leaves simtools itself to be installed from a
bind-mounted checkout.

Run a development image with

```console
podman run --rm -it \
  -v "$(pwd):/workdir/external/simtools" \
  ghcr.io/gammasim/simtools-dev:latest \
  bash -lc "cd /workdir/external/simtools && pip install -e . && exec bash"
```

Published production images include `/opt/simtools/provenance/dependency-manifest.json`. The
production workflow converts each pushed OCI image to SIF from its digest, verifies the manifest,
and publishes the SIF through GHCR using the `-apptainer` tag suffix.

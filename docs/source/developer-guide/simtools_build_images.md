# Build Images

Pre-built OCI and Apptainer images are available from the
[simtools package registry](https://github.com/orgs/gammasim/packages?repo_name=simtools).
The GitHub Actions workflows in `.github/workflows/build-*.yml` are the reference image builds.

All build versions come from `pyproject.toml`; see
[Dependency versions and provenance](dependency_versions.md). Dockerfiles deliberately have no
independent software-version defaults. Before a local build, export the validated values with

```console
python src/simtools/dependency_versions.py --format github-output
```

## Scientific component images

`docker/Dockerfile-corsika7` builds each catalogued CORSIKA and CPU variant. It requires the CORSIKA
source token and the `autoconf.tar.gz` archive. The Dockerfile verifies the catalogued source,
configuration and optimization-patch revisions plus the archive checksum.

`docker/Dockerfile-simtel_array` builds the catalogued sim_telarray, hessio and stdtools revisions.
It requires the corresponding GitLab tokens and `gsl.tar.gz`; the archive checksum is verified
before extraction.

Use the workflow-generated matrix values as build arguments. This ensures that a local build uses
the same base-image digests, source revisions and flags as CI.

## Production and development images

`docker/Dockerfile-simtools-prod` installs the checked-out simtools revision with the compatible
Python dependencies declared in `pyproject.toml`. Its CORSIKA, sim_telarray and AlmaLinux inputs
are immutable OCI digest references.

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

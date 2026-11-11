# Build Images

Pre-built OCI images are available from the
[simtools package registry](https://github.com/orgs/gammasim/packages?repo_name=simtools).
The GitHub Actions workflows in `.github/workflows/build-simtools-*.yml` build the simtools images.

## Build versions

Simtools image build versions come from `dependency_versions.yml`; see
[Dependency versions and provenance](dependency_versions.md). Dockerfiles provide fallback values
for standalone local builds, while the workflows pass the values generated from the catalog.
Before a reproducible local build, export the validated values with

```console
simtools-dependency-versions --format github-output
```

## Scientific component images

The sim_telarray (including hessio and stdtools) and CORSIKA images are built in the GitLab CI of
the sim_telarray and corsika7-config repositories. They are published as
`ghcr.io/gammasim/sim_telarray` and `ghcr.io/gammasim/corsika7`. The simtools images consume them
through the `SIMTEL_IMAGE` and `CORSIKA_IMAGE` build arguments.

## Production and development images

`docker/Dockerfile-simtools-prod` installs the checked-out simtools revision with the compatible
Python dependencies declared in `pyproject.toml`. Its CORSIKA, sim_telarray and AlmaLinux inputs
are version-tag references unless optional OCI digests are declared.

`docker/Dockerfile-simtools-dev` installs the same compatible Python dependencies, including the
development, documentation and test extras, but leaves simtools itself to be installed from a
bind-mounted checkout. Its default user has UID and GID 1000. Set the `CONTAINER_UID` and
`CONTAINER_GID` build arguments to match the owner of a bind-mounted checkout when building a
local image.

Run a development image with

```console
podman run --rm -it \
  -v "$(pwd):/workdir/external/simtools" \
  ghcr.io/gammasim/simtools-dev:latest \
  bash -lc "cd /workdir/external/simtools && pip install -e . && exec bash"
```

When using a published image whose UID and GID do not match the checkout owner, run it with
`--user "$(id -u):$(id -g)" -e HOME=/tmp` so the mounted checkout and user-level tools remain
writable.

Published production images include `/opt/simtools/provenance/dependency-manifest.json`. Apptainer
users should pull a production image directly from the OCI registry with a `docker://` reference.

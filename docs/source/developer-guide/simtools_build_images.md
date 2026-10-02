# Build Images

Pre-built OCI images are available from the
[simtools package registry](https://github.com/orgs/gammasim/packages?repo_name=simtools).
The GitHub Actions workflows in `.github/workflows/build-*.yml` are the reference image builds.

## Build dependency graph

```text
dependency_versions.yml
        |
        +-- weekly refresh: compare remote commits, build snapshots, open review PR
        |                                                           |
        v                                                           v
  private CORSIKA source snapshot                           private sim_telarray source snapshot
  (source and patches)                                      (source and dependencies)
        |                                                           |
        +---------------- GitHub Actions artifacts ----------------+
                                                    |
                                                    v
                         CORSIKA and sim_telarray image builds (per architecture)
                                                    |
                                                    v
                                      GHCR scientific component images
                                                    |
                         +--------------------------+--------------------------+
                         v                                                     v
              simtools production image                                  simtools dev image
                         |
                         v
              reusable integration-test workflow
```

All scientific build versions come from `dependency_versions.yml`; see
[Dependency versions and provenance](dependency_versions.md). Dockerfiles provide fallback values
for standalone local builds, while the workflows pass the values generated from the catalog.
Before a reproducible local build, export the validated values with

```console
simtools-dependency-versions --format github-output
```

## Scientific component images

`docker/Dockerfile-corsika7` builds each catalogued CORSIKA and CPU variant. The workflow prepares
the CORSIKA source, configuration, and optimization-patch trees before the Docker build and
provides them with the separately downloaded `autoconf.tar.gz` archive. Once the catalog contains
an immutable source snapshot digest, the source inputs are restored from private GHCR instead of
fetched from remote sites. The archive remains outside the snapshot and uses its verified download
fallback.

`docker/Dockerfile-simtel_array` builds the catalogued sim_telarray, hessio, and stdtools releases.
The workflow prepares those source trees before the Docker build and provides them with the
separately downloaded `gsl.tar.gz` archive. Once the catalog contains an immutable source snapshot
digest, the source inputs are restored from private GHCR instead of fetched from remote sites. The
archive remains outside the snapshot and uses its verified download fallback.

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

Published production images include `/opt/simtools/provenance/dependency-manifest.json`. Apptainer
users should pull a production image directly from the OCI registry with a `docker://` reference.

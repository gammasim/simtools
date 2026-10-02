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
  private CORSIKA input snapshot                            private sim_telarray input snapshot
  (source, patches, Autoconf)                               (source, dependencies, GSL)
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

## Refreshing source snapshots

`Refresh build-input snapshots` runs weekly on Monday at 03:00 UTC and can also be started with
**Run workflow** from the GitHub Actions page. It resolves the configured source references and
builds a new private GHCR snapshot only when a resolved source commit has changed.

When a new snapshot is published, the workflow opens or updates a review PR. The PR is labelled
`no-changelog-needed` and changes only `dependency_versions.yml`: the resolved source revisions
and private snapshot digest. It never adds the generated source or auxiliary archives to Git.

## Scientific component images

`docker/Dockerfile-corsika7` builds each catalogued CORSIKA and CPU variant. The workflow prepares
the CORSIKA source, configuration, and optimization-patch trees before the Docker build and
provides them with the `autoconf.tar.gz` archive. Once the catalog contains an immutable source
snapshot digest, these inputs are restored from private GHCR instead of fetched from remote sites.

`docker/Dockerfile-simtel_array` builds the catalogued sim_telarray, hessio, and stdtools releases.
The workflow prepares those source trees before the Docker build and provides them with the
`gsl.tar.gz` archive. Once the catalog contains an immutable source snapshot digest, these inputs
are restored from private GHCR instead of fetched from remote sites.

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

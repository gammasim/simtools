# Containers

Container files are available for [simtools](https://github.com/gammasim/simtools) for both development and runtime use. Pre-built container images are available through the [simtools package registry](https://github.com/orgs/gammasim/packages?repo_name=simtools).

See the [simtools container documentation](https://gammasim.github.io/simtools/developer-guide/simtools_build_images.html) for build details and the [dependency version documentation](https://gammasim.github.io/simtools/developer-guide/dependency_versions.html) for the version catalog and provenance manifest.

For convenience, the command to use a container as a developer is repeated here:

```bash
podman run --rm -it -v "$(pwd)/:/workdir/external" ghcr.io/gammasim/simtools-dev:latest bash -lc "source /workdir/env/bin/activate && cd /workdir/external/simtools && pip install -e . && bash"
```

In case a local database is used:

```bash
podman run --rm -it -v "$(pwd)/:/workdir/external" --network simtools-mongo-network ghcr.io/gammasim/simtools-dev:latest bash -lc "source /workdir/env/bin/activate && cd /workdir/external/simtools && pip install -e . && bash"
```

In some combinations of local git directory and container settings, git might report an issue with unsafe directories. In this case, run:

```bash
podman run --rm -it -v "$(pwd)/:/workdir/external" --network simtools-mongo-network ghcr.io/gammasim/simtools-dev:latest bash -c "git config --global --add safe.directory /workdir/external/simtools && cd /workdir/external/simtools && pip install -e . && exec bash"
 ```

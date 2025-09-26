# Containers

Container files are available for [simtools](https://github.com/gammasim/simtools) for both development and applications. Pre-built container images are available through the [simtools package registry](https://github.com/orgs/gammasim/packages?repo_name=simtools).

See the [simtools container documentation](https://gammasim.github.io/simtools/user-guide/docker_files.html) for further details:

For convenience, the command to use a container as developer is repeated here:

```bash
podman run --rm -it -v "$(pwd)/:/workdir/external" ghcr.io/gammasim/simtools-dev:latest bash -c "source /workdir/env/bin/activate && cd /workdir/external/simtools && pip install -e . && bash"
```

In case a local database is used:

```bash
podman run --rm -it -v "$(pwd)/:/workdir/external" --network simtools-mongo-network ghcr.io/gammasim/simtools-dev:latest bash -c "source /workdir/env/bin/activate && cd /workdir/external/simtools && pip install -e . && bash"
```

# Build Images

Pre-built images are available from the [simtools package registry](https://github.com/orgs/gammasim/packages?repo_name=simtools).
Replace the docker command by the container run engine of your choice (e.g., podman).

## Build a simulation production container

To build an image locally run:

```bash
docker build -f Dockerfile-prod-opt -t simtools .
```

The build process requires a tarball of corsika/sim\_telarray (named `corsika_simtelarray.tar.gz`) to be present in the build directory.
This package is available from MPIK (password protected).
Build arguments can be configured as specified at the top of the Dockerfile.

Run the newly built container:

```bash
docker run --rm -it -v "$(pwd)/external:/workdir/external" simtools bash
```

## Build a developers container locally

To build an image locally run in the [docker](https://github.com/gammasim/simtools/tree/main/docker) directory:

```bash
docker build -f Dockerfile-dev -t simtools-dev .
```

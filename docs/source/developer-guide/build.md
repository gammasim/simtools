# Building simtools

Users and developers typically do not need to build simtools from source. Instead, pre-built images
from the [simtools package registry](https://github.com/orgs/gammasim/packages?repo_name=simtools) should be used.
The page below gives some incomplete information on how to build simtools from source.

## Build a new simulation production container locally

To build a new container locally run:

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

## Build a new developers container locally

To build a new container locally run in the [docker](https://github.com/gammasim/simtools/tree/main/docker) directory:

```bash
docker build -f Dockerfile-dev -t simtools-dev .
```

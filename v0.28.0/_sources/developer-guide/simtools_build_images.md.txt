# Build Images

Pre-built images are available from the [simtools package registry](https://github.com/orgs/gammasim/packages?repo_name=simtools).
Replace the command by the container run engine of your choice (e.g., Docker).

## Build a simulation production container

First build the CORSIKA/sim_telarray container locally (or use the pre-built image from the package registry):

```bash
podman buildx build --platform=linux/arm64 --secret id=gitlab_token,src=./my_secret --build-arg AVX_FLAG=generic -f Dockerfile-corsika-simtel -t corsika-simtelarray .
```

(requires a secret token to access the CTAO GitLab repository).

Then build the simtools production container:

```bash
podman buildx build --platform=linux/arm64 --build-arg BASE_IMAGE=localhost/corsika-simtelarray --build-arg BUILD_BRANCH=main --build-arg PYTHON_VERSION=3.12 -f Dockerfile-simtools -t simtools  .
```

The build process requires a tarball of corsika/sim\_telarray (named `corsika_simtelarray.tar.gz`) to be present in the build directory.
This package is available from MPIK (password protected).
Build arguments can be configured as specified at the top of the Dockerfile.

Run the newly built container:

```bash
podman run --rm -it -v "$(pwd)/external:/workdir/external" simtools bash
```

## Build a developers container locally

To build an image locally run in the [`./docker`](https://github.com/gammasim/simtools/tree/main/docker) directory:

```bash
podman build -f Dockerfile-dev -t simtools-dev .
```

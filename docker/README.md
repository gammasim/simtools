# Docker files for simtools

Docker files for [simtools](https://github.com/gammasim/simtools) development and applications.

[Docker](https://www.docker.com/community-edition#/download) or any compatible software are required to run or build these images.

Types of dockerfiles and containers available:

- simtools users: [Dockerfile-prod](./Dockerfile-prod) provides a container with all software installed (CORSIKA, sim\_telarray, simtools python environment, simtools). Pull latest release with: `docker pull ghcr.io/gammasim/simtools-prod:latest`
- simtools developers: [Dockerfile-dev](./Dockerfile-dev) provides a container with CORSIKA, sim\_telarray, and simtools conda environment installed. Pull latest release with: `docker pull ghcr.io/gammasim/simtools-dev:latest`
- sim\_telarray: [Dockerfile-simtelarray](./Dockerfile-simtelarray) provides a container with the CORSIKA and sim\_telarray installed. This provides the base image for the previously listed containers. Pull latest release with: `docker pull ghcr.io/gammasim/simtools-simtelarray:latest`

## Container for simtools users

Provide a container for simtools users.

Container includes installation of:

- corsika and sim\_telarray
- packages required by simtools (from pyproject.toml)
- simtools (main branch)

Images are automatically built by the [Github action workflow ../.github/workflows/build-docker-image.yml](../.github/workflows/build-docker-image.yml) and can be downloaded from the [gammasim package website](https://github.com/gammasim/containers/pkgs/container/simtools-prod). **TODO - fix path**

### Run a simtools-prod container

To run the container in bash

```bash
docker run ghcr.io/gammasim/simtools-prod:latest bash
```

In the container, simtools applications are installed and can be called directly (e.g., `simtools-print-array-elements -h`).

In case file exchange with the local file system is required, use the docker syntax to mount a directory. Example:

```bash
docker run --rm -it -v "$(pwd):/workdir/external" ghcr.io/gammasim/simtools-prod:latest bash
```

The following example runs an application inside the container and write the output into a directory of the local files system,

```bash
docker run --rm -it -v "$(pwd):/workdir/external" \
    ghcr.io/gammasim/simtools-prod:latest \
    simtools-print-array-elements \
    --array_element_list ./simtools/tests/resources/telescope_positions-North-utm.ecsv \
    --export corsika --use_corsika_telescope_height \
    --output_path /workdir/external/
```

Output files can be found `./simtools-output/`.

### Building a simtools-prod container

To build a new container locally run:

```bash
docker build -f Dockerfile-prod  -t simtools-prod .
```

Building will take a while and the image is large (~1.4 GB). For using images build on your own, replace in all examples `ghcr.io/gammasim/simtools-prod:latest` by the local image name `simtools-prod`.

## Container for simtools developers

Provide a container for testing and development of simtools. This container is not optimised for size, but for completeness of development tools.

Container includes installation of:

- corsika and sim\_telarray
- packages required by simtools (from pyproject.toml)

The container does not include the simtools code, which should be cloned in a separate directory (see below).

There are two options on how to use this container:

1. Download from [simtools container repository](https://github.com/gammasim/containers/pkgs/container/simtools-dev) **TODO - fix link**
2. Build a new container from the available Dockerfile (requires access to sim\_telarray package)

### Run a container using the prepared Docker image available from repository

Packages are available from the [simtools container repository](https://github.com/gammasim/containers/pkgs/container/simtools-dev)**TODO**

**Preparation:**

Create a new directory for your development and clone simtools into a subdirectory:

```bash
mkdir -p simtools-dev && simtools-dev
git clone git@github.com:gammasim/simtools.git
```

To download and run a prepared container in bash:

```bash
docker run --rm -it -v "$(pwd)/:/workdir/external" ghcr.io/gammasim/simtools-dev:latest bash -c "$(cat ./simtools/docker/entrypoint.sh) && bash"
```

This additionally executes the `entrypoint.sh` script (e.g., for pip install or set the database environment).

Remember you need to `docker login` to the GitHub package repository with a personal token in order to download an image (follow [these instructions](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)).

### Build a new developers container locally

To build a new container locally run:

```bash
docker build -f Dockerfile-dev -t simtools-dev .
```

Use the docker container in the same way as above, replacing `ghcr.io/gammasim/simtools-dev:latest` by `simtools-dev`.

## Container for simulation software CORSIKA / sim\_telarray

Provide a container including the following the CORSIKA and sim\_telarray simulation software packages.

There are two options on how to use this container:

1. Download from [simtools container repository](https://github.com/gammasim/containers/pkgs/container/simtools-simtel)
2. Build a new container from the available [Dockerfile](./Dockerfile-simtelarray) (requires access to sim\_telarray package)

### Download from simtools container repository

Packages are available from the [simtools container repository](https://github.com/gammasim/containers/pkgs/container/simtools-simtel).

To download and run a prepared container in bash:

```bash
docker run --rm -it -v "$(pwd)/external:/workdir/external" ghcr.io/gammasim/simtools-simtelarray:latest bash
```

## Build a new container locally

To build a new container locally run:

```bash
docker build -f Dockerfile-simtelarray  -t sim_telarray .
```

Building expects that a tar ball of corsika/sim\_telarray (named corsika7.7\_simtelarray.tar.gz) is available in the building directory.
Download the tar package from the MPIK website (password applies) with

```bash
./download_simulationsoftware.sh
```

Run the newly build container:

```bash
docker run --rm -it -v "$(pwd)/external:/workdir/external" simtelarray bash
```

**Apple silicon users, notice you should add --platform=linux/amd64 to the run command above.**

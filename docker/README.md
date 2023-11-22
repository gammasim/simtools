# Docker files for simtools

Docker files for [simtools](https://github.com/gammasim/simtools) development and applications. Prepared packages are available through the [simtools package registry](https://github.com/orgs/gammasim/packages?repo_name=simtools).

[Docker](https://www.docker.com/community-edition#/download) or any compatible software are required to run or build these images.

Types of dockerfiles and containers available:

- [simtools users](#container-for-simtools-users): a container with all software installed (CORSIKA, sim\_telarray, simtools python environment, simtools). Pull latest release with: `docker pull ghcr.io/gammasim/simtools-prod:latest`
- [simtools developers](#container-for-simtools-developers): a container with CORSIKA, sim\_telarray, and simtools conda environment installed. Pull latest release with: `docker pull ghcr.io/gammasim/simtools-dev:latest`
- [sim_telarray](#container-for-corsika-and-simtelarray): provides a container with the CORSIKA and sim\_telarray installed. This provides the base image for the previously listed containers. Pull latest release with: `docker pull ghcr.io/gammasim/simtools-simtelarray:latest`

## Container for simtools Users

Provide a container for simtools users, which includes:

- corsika and sim\_telarray
- packages required by simtools (from pyproject.toml)
- simtools (main branch)

### Run a simtools-prod container

Prepare a file for the simulation model database access (see simtools documentation) similar to the [template example](https://github.com/gammasim/simtools/blob/main/.env_template) provided with simtools.

To run the container in bash

```bash
docker run --rm -it --env-file .env -v "$(pwd):/workdir/external" ghcr.io/gammasim/simtools-prod:latest bash
```

In the container, simtools applications are installed and can be called directly (e.g., `simtools-print-array-elements -h`).
This example uses the docker syntax to mount your local directory.

The following example runs an application inside the container and writes the output into a directory of the local files system,

```bash
docker run --rm -it --env-file .env \
    -v "$(pwd):/workdir/external" \
    ghcr.io/gammasim/simtools-prod:latest \
    simtools-print-array-elements \
    --array_element_list ./simtools/tests/resources/telescope_positions-North-utm.ecsv \
    --export corsika --use_corsika_telescope_height \
    --output_path /workdir/external/
```

Output files can be found `./simtools-output/`.

### Building a simtools-prod container

To build a new container locally run in the [simtools/docker](simtools/docker) directory::

```bash
docker build -f Dockerfile-prod  -t simtools-prod .
```

Building will take a while and the image is large (~1.4 GB). For using images build on your own, replace in all examples `ghcr.io/gammasim/simtools-prod:latest` by the local image name `simtools-prod`.

## Container for simtools Developers

Provide a container for testing and development of simtools, including:

- CORSIKA and sim\_telarray
- packages required by simtools (from pyproject.toml)

The container does not include the simtools code, which should be cloned in a separate directory (see below).

### Run a container using the prepared Docker image available from repository

Packages are available from the [simtools container repository](https://github.com/orgs/gammasim/packages?repo_name=simtools).

**Preparation:**

Create a new directory for your development and clone simtools into a subdirectory:

```bash
mkdir -p simtools-dev && cd simtools-dev
git clone git@github.com:gammasim/simtools.git
```

To download and run a prepared container in bash:

```bash
docker run --rm -it -v "$(pwd)/:/workdir/external" ghcr.io/gammasim/simtools-dev:latest bash -c "source /workdir/env/bin/activate && cd /workdir/external/simtools && pip install -e . && bash"
```

Remember you need to `docker login` to the GitHub package repository with a personal token in order to download an image (follow [these instructions](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)).

### Build a new developers container locally

To build a new container locally run in the [simtools/docker](simtools/docker) directory:

```bash
docker build -f Dockerfile-dev -t simtools-dev .
```

Use the docker container in the same way as above, replacing `ghcr.io/gammasim/simtools-dev:latest` by `simtools-dev`.

## Container for CORSIKA and sim\_telarray

Provide a container including the following the CORSIKA and sim\_telarray simulation software packages.

### Download from simtools container repository

Packages are available from the [simtools container repository](https://github.com/gammasim/containers/pkgs/container/simtools-simtel).

To download and run a prepared container in bash:

```bash
docker run --rm -it -v "$(pwd)/external:/workdir/external" ghcr.io/gammasim/simtools-simtelarray:latest bash
```

## Build a new container locally

To build a new container locally run:

```bash
docker build -f Dockerfile-simtelarray  -t simtelarray .
```

Building expects that a tar ball of corsika/sim\_telarray (named corsika7.7\_simtelarray.tar.gz) is available in the building directory.
Download the tar package from MPIK (password applies) with

```bash
./download_simulationsoftware.sh
```

Run the newly build container:

```bash
docker run --rm -it -v "$(pwd)/external:/workdir/external" simtelarray bash
```

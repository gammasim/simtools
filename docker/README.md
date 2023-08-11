# Docker files for simtools

Docker files for [simtools](https://github.com/gammasim/simtools) development and applications.

[Docker](https://www.docker.com/community-edition#/download) or any compatible software are required to run these images.

Types of dockerfiles and containers available:

- simtools users: [Dockerfile-prod](./Dockerfile-prod) provides a container with all software installed (CORSIKA, sim\_telarray, simtools python environment, simtools). Pull latest release with: `docker pull ghcr.io/gammasim/simtools-prod:latest`
- simtools development: [Dockerfile-dev](./Dockerfile-dev) provides a container with CORSIKA, sim\_telarray, and simtools conda environment installed. Pull latest release with: `docker pull ghcr.io/gammasim/simtools-dev:latest`
- sim\_telarray: [Dockerfile-simtelarray](./Dockerfile-simtelarray) provides a container with the CORSIKA and sim\_telarray installed. Pull latest release with: `docker pull ghcr.io/gammasim/simtools-simtelarray:latest`

The CORSIKA / sim\_telarray packages can be downloaded from MPIK (authentication required).

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

__Apple silicon users, notice you should add --platform=linux/amd64 to the run command above.__

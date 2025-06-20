# Running simtools Containers

Container images are available for [simtools](https://github.com/gammasim/simtools) for both development and applications through the [simtools package registry](https://github.com/orgs/gammasim/packages?repo_name=simtools).
[Podman](https://podman.io/), [apptainer](https://apptainer.org/), [Docker](https://www.docker.com/community-edition#/download) or any compatible software can be used to run the containers.

Types of simtools images:

- **simtools production images**: are images used for the production simulations, which include CORSIKA, sim_telarray, and simtools applications. These images are built:
  - with different stable CORSIKA and sim_telarray versions
  - with different compile options (e.g., `prod5`, `prod6`)
  - for different CPU vector-code optimizations (e.g., avx2, avx512). Images with `no_opt` tag are built without vector-code optimizations.
- **simtools development images**: these are the images used for simtools development, which include CORSIKA, sim_telarray, and the environment including all packages required to run simtools (but not simtools itself).

## Simtools package registry

Pre-built container images are available through the [simtools package registry](https://github.com/orgs/gammasim/packages?repo_name=simtools).

Follow the [instruction](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry) (`docker login`) to authenticate with the GitHub package registry before pulling the first image.

## Run a simtools production image (simtools-prod)

Prerequisite: configure the simulation model database access (see simtools documentation) similar to the [template example](https://github.com/gammasim/simtools/blob/main/.env_template).

To startup a container to use bash and mount your local directory for file access:

```bash
docker run --rm -it --env-file .env -v "$(pwd):/workdir/external" ghcr.io/gammasim/simtools-prod-240205-corsika-77500-bernlohr-1.67-prod6-baseline-qgs2-no_opt:latest bash
```

In the container, simtools applications are installed and can be called directly.

The following example runs an application inside the container and writes the output into a directory of the local files system,

```bash
docker run --rm -it --env-file .env \
    -v "$(pwd):/workdir/external" \
    ghcr.io/gammasim/simtools-prod-240205-corsika-77500-bernlohr-1.67-prod6-baseline-qgs2-no_opt:latest \
    simtools-convert-geo-coordinates-of-array-elements \
    --input ./simtools/tests/resources/telescope_positions-North-utm.ecsv \
    --export ground \
    --output_path /workdir/external/
```

Output files can be found `./simtools-output/`.

## Container for simtools developers (simtools-dev)

**Preparation:**

Create a new directory for your development and clone simtools into a subdirectory:

```bash
mkdir -p simtools-dev && cd simtools-dev
git clone git@github.com:gammasim/simtools.git
```

To download and run a prepared container with bash:

```bash
docker run --rm -it -v "$(pwd)/:/workdir/external" ghcr.io/gammasim/simtools-dev:latest bash -c "source /workdir/env/bin/activate && cd /workdir/external/simtools && pip install -e . && bash"
```

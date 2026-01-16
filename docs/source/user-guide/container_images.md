# Container Images

OCI-compatible container images are available for simtools users and support both application and development use cases. Runtimes such as [Docker](https://www.docker.com/products/docker-desktop), [Podman](https://podman.io/), or [Apptainer](https://apptainer.org/) can be used.

## Pre-built Images

Below an overview of the most-relevant pre-built container images available from the [simtools package registry](https://github.com/orgs/gammasim/packages?repo_name=simtools).

All images use `almalinux:9.5` or `almalinux:9.5-minimal` with python 3.12 as base images.

### Simtools for Production

- registry page: [simtools-prod](https://github.com/gammasim/simtools/pkgs/container/simtools-prod)
- docker file: [Dockerfile-simtools-prod](https://github.com/gammasim/simtools/blob/main/docker/Dockerfile-simtools-prod)

| image tag | simtools | CORSIKA | sim_telarray |
| --- | --- | --- | --- |
| 0.27.0-[generic](ghcr.io/gammasim/simtools-prod:0.27.0-generic), [avx2](ghcr.io/gammasim/simtools-prod:0.27.0-avx2), [avx512f](ghcr.io/gammasim/simtools-prod:0.27.0-avx512f), [sse4](ghcr.io/gammasim/simtools-prod:0.27.0-sse4) | 0.27.0 | v78010 | v2025-11-30-rc |

The generic image is compatible with ARM and x86_64 CPUs, all other images are for x86_64 CPUs only.

### Simtools for Developers

- registry page: [simtools-dev](https://github.com/gammasim/simtools/pkgs/container/simtools-dev)
- docker file: [Dockerfile-simtools-dev](https://github.com/gammasim/simtools/blob/main/docker/Dockerfile-simtools-dev)

| image tag | simtools | CORSIKA | sim_telarray | remarks |
| --- | --- | --- | --- | --- |
| [0.27.0](ghcr.io/gammasim/simtools-dev:0.27.0) | 0.27.0 | 7.8010 | 2025-11-30-rc | includes QGSJet3 tables |
| [0.26.0](ghcr.io/gammasim/simtools-dev:0.26.0) | 0.26.0 | 7.8010 | master | includes QGSJet3 tables |
| [0.25.0](ghcr.io/gammasim/simtools-dev:0.25.0) | 0.25.0 | 7.8010 | master | includes QGSJet3 tables  |

Developments are compatible with ARM and x86_64 CPUs.

### CORSIKA7

- registry page: [corsika7](https://github.com/gammasim/simtools/pkgs/container/corsika7)
- docker file: [Dockerfile-corsika7](https://github.com/gammasim/simtools/blob/main/docker/Dockerfile-corsika7)

| image tag | CORSIKA | high-/low-energy <br>interaction models | compile <br>configuration | optimization <br> patches | IACT/ATMO |
| --- | --- | --- | --- | --- | --- |
| v78010-[generic](ghcr.io/gammasim/corsika7:v78010-generic), [avx2](ghcr.io/gammasim/corsika7:v78010-avx2), [avx512f](ghcr.io/gammasim/corsika7:v78010-avx512f), [sse4](ghcr.io/gammasim/corsika7:v78010-sse4) | 7.8010 | QGSJet-III/URQMD; EPOS/URQMD | [v0.1.0](https://gitlab.cta-observatory.org/cta-computing/dpps/simpipe/simulation_software/corsika7-config) | [v1.1.0](https://gitlab.cta-observatory.org/cta-computing/dpps/simpipe/simulation_software/corsika-opt-patches/-/releases/v1.1.0) | 1.69 |

The `generic` image is compatible with ARM and x86_64 CPUs, all other images are for x86_64 CPUs only.

### sim_telarray Images

- registry page: [sim_telarray](https://github.com/gammasim/simtools/pkgs/container/sim_telarray)
- docker file: [Dockerfile-sim_telarray](https://github.com/gammasim/simtools/blob/main/docker/Dockerfile-simtel_array)

| Image tag | sim_telarray | hessio version | stdtools version |
| --- | --- | --- | --- |
v2025-11-30-rc | [v2025-11-30-rc](https://gitlab.cta-observatory.org/cta-computing/dpps/simpipe/simulation_software/sim_telarray) | [v2025-12-01-rc](https://gitlab.cta-observatory.org/cta-computing/dpps/simpipe/simulation_software/hessio) | [v2025-06-16-rc](https://gitlab.cta-observatory.org/cta-computing/dpps/simpipe/simulation_software/stdtools) |

Note: Version names should be replaced with actual version numbers when releases are available.

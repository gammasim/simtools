# Container Images

OCI-compatible container images are available for simtools users and support both application and development use cases. Runtimes such as [Docker](https://www.docker.com/products/docker-desktop), [Podman](https://podman.io/), or [Apptainer](https://apptainer.org/) can be used.

## Pre-built Images

Below an overview of the most-relevant pre-built container images available from the [simtools package registry](https://github.com/orgs/gammasim/packages?repo_name=simtools).

### Simtools Production

- registry page: [simtools-prod](https://github.com/gammasim/simtools/pkgs/container/simtools-prod)
- docker file: [Dockerfile-simtools-prod](https://github.com/gammasim/simtools/blob/main/docker/Dockerfile-simtools-prod)

| Image tag | simtools | CORSIKA | sim_telarray | AVX optimizations |
| --- | --- | --- | --- | --- |
| [0.26.0-corsika-20251217-78010-no_opt-simtel-latest](ghcr.io/gammasim/simtools-prod:0.26.0-corsika-20251217-78010-no_opt-simtel-latest) | 0.26.0 | 20251217-78010 | latest | no_opt |
| [0.26.0-corsika-20251217-78010-avx2-simtel-latest](ghcr.io/gammasim/simtools-prod:0.26.0-corsika-20251217-78010-avx2-simtel-latest) | 0.26.0 | 20251217-78010 | latest | avx2 |
| [0.26.0-corsika-20251217-78010-avx512f-simtel-latest](ghcr.io/gammasim/simtools-prod:0.26.0-corsika-20251217-78010-avx512f-simtel-latest) | 0.26.0 | 20251217-78010 | latest | avx512f |
| [0.26.0-corsika-20251217-78010-sse4-simtel-latest](ghcr.io/gammasim/simtools-prod:0.26.0-corsika-20251217-78010-sse4-simtel-latest) | 0.26.0 | 20251217-78010 | latest | sse4 |

The no_opt image is compatible with ARM and x86_64 CPUs, all other images are for x86_64 CPUs only.

### Simtools for Developers

- registry page: [simtools-dev](https://github.com/gammasim/simtools/pkgs/container/simtools-dev)
- docker file: [Dockerfile-simtools-dev](https://github.com/gammasim/simtools/blob/main/docker/Dockerfile-simtools-dev)

| Image tag | simtools | CORSIKA | sim_telarray | remarks |
| --- | --- | --- | --- | --- |
| [0.26.0](ghcr.io/gammasim/simtools-dev:0.26.0) | 0.26.0 | 7.8010 | latest | includes QGSJet3 tables |
| [0.25.0](ghcr.io/gammasim/simtools-dev:0.25.0) | 0.25.0 | 7.8010 | latest | includes QGSJet3 tables  |

Developments are compatible with ARM and x86_64 CPUs.

### CORSIKA7

- registry page: [corsika7](https://github.com/gammasim/simtools/pkgs/container/corsika7)
- docker file: [Dockerfile-corsika7](https://github.com/gammasim/simtools/blob/main/docker/Dockerfile-corsika7)

| Image tag | CORSIKA | high-/low-energy interaction models | Compile Configuration | CORSIKA Patch (optimizations) | IACT/ATMO |
| --- | --- | --- | --- | --- | --- |
20251217-78010 | 7.8010 | QGSJet-III/URQMD; EPOS/URQMD | [v0.1.0](https://gitlab.cta-observatory.org/cta-computing/dpps/simpipe/simulation_software/corsika7-config) | [v1.1.0](https://gitlab.cta-observatory.org/cta-computing/dpps/simpipe/simulation_software/corsika-opt-patches/-/releases/v1.1.0) | 1.69 |

Images are provided for the following CPU optimizations (example links for version 20251217-78010): [no optimization](ghcr.io/gammasim/corsika7:20251217-78010-no_opt), [avx2]( ghcr.io/gammasim/corsika7:20251217-78010-avx2), [avx512f]( ghcr.io/gammasim/corsika7:20251217-78010-avx512f), [sse4](ghcr.io/gammasim/corsika7:20251217-78010-sse4). The no_opt image is compatible with ARM and x86_64 CPUs, all other images are for x86_64 CPUs only.

### sim_telarray Images

- registry page: [sim_telarray](https://github.com/gammasim/simtools/pkgs/container/sim_telarray)
- docker file: [Dockerfile-sim_telarray](https://github.com/gammasim/simtools/blob/main/docker/Dockerfile-simtel_array)

| Image tag | sim_telarray | hessio version | stdtools version |
| --- | --- | --- | --- |
20251217-master| [master](https://gitlab.cta-observatory.org/cta-computing/dpps/simpipe/simulation_software/sim_telarray) | [master](https://gitlab.cta-observatory.org/cta-computing/dpps/simpipe/simulation_software/hessio) | [master](https://gitlab.cta-observatory.org/cta-computing/dpps/simpipe/simulation_software/stdtools) |

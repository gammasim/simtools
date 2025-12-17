# Container Images

OCI-compatible container images are available for simtools users and support both application and development use cases.  Any runtime such as [Docker](https://www.docker.com/products/docker-desktop), [Podman](https://podman.io/), or [Apptainer](https://apptainer.org/) can be used.

## Pre-built Images

Below an overview of the most-relevant pre-built container images available from the [simtools package registry](https://github.com/orgs/gammasim/packages?repo_name=simtools).

### Simtools Production Images

[Dockerfile-simtools-prod](https://github.com/gammasim/simtools/blob/main/docker/Dockerfile-simtools-prod)

| Image tag | simtools version | CORSIKA image tag | sim_telarray image tag |
| --- | --- | --- | --- |
| latest | 0.26.0 | 20251216-095556 | 20251210-112343 |

#### Legacy Simtools Production Images

[Dockerfile-simtools-legacy](https://github.com/gammasim/simtools/blob/main/docker/Dockerfile-simtools-legacy)

### Simtools Development Images

[Dockerfile-simtools-dev](https://github.com/gammasim/simtools/blob/main/docker/Dockerfile-simtools-dev)

package | simtools version | CORSIKA (patch) version | high-/low-energy interaction model | sim_telarray (IACT) version | remarks |
| --- | --- | --- | --- | --- | --- |
| [20251216-102359](ghcr.io/gammasim/simtools-dev:20251216-102359) | 0.26.0  | 7.8010 (v1.1.0) | QGSJet-III/URQMD | 2025.246.0 (1.70) | CORSIKA/sim_telarray from gitlab (sim_telarray from master branch) |
| [0.25.0](ghcr.io/gammasim/simtools-dev:0.25.0) | 0.25.0 | 7.8010 (v1.1.0) | QGSJet-III/URQMD | 2025.246.0 (1.70) | CORSIKA/sim_telarray installation from tar packages |

### CORSIKA7 Images

[Dockerfile-corsika7](https://github.com/gammasim/simtools/blob/main/docker/Dockerfile-corsika7)

- [ ] TODO: release compile configurations and update link
- [ ] TODO: change tag to released version

| Image tag | CORSIKA version | high-/low-energy interaction model | Compile Configuration | Patch version (CPU optimizations) | Packages |
| --- | --- | --- | --- | --- | --- |
20251216-095556 | 7.8010 | QGSJet-III/URQMD | [v0.1.0](https://gitlab.cta-observatory.org/cta-computing/dpps/simpipe/simulation_software/corsika7-config) | [v1.1.0](https://gitlab.cta-observatory.org/cta-computing/dpps/simpipe/simulation_software/corsika-opt-patches/-/releases/v1.1.0) | [no optimization](https://github.com/gammasim/simtools/pkgs/container/corsika7-78010-no_opt), [AVX2](https://github.com/gammasim/simtools/pkgs/container/corsika7-78010-avx2), [AVX512](https://github.com/gammasim/simtools/pkgs/container/corsika7-78010-avx512f), [sse4](https://github.com/gammasim/simtools/pkgs/container/corsika7-78010-sse4)|

### sim_telarray Images

[Dockerfile-sim_telarray](https://github.com/gammasim/simtools/blob/main/docker/Dockerfile-simtel_array)

- [ ] TODO: assign version numbers to sim_telarray packages and release
- [ ] TODO: change tag to released version

Image tag | sim_telarray version | hessio version | stdtools version | Packages |
| --- | --- | --- | --- | --- |
20251210-112343 | [master](https://gitlab.cta-observatory.org/cta-computing/dpps/simpipe/simulation_software/sim_telarray) | [master](https://gitlab.cta-observatory.org/cta-computing/dpps/simpipe/simulation_software/hessio) | [master](https://gitlab.cta-observatory.org/cta-computing/dpps/simpipe/simulation_software/stdtools) | [prod6-baseline](https://github.com/gammasim/simtools/pkgs/container/simtel-251124-prod6-baseline)

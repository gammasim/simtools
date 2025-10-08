# Containers

Container files are available for [simtools](https://github.com/gammasim/simtools) for both development and applications. Pre-built container images are available through the [simtools package registry](https://github.com/orgs/gammasim/packages?repo_name=simtools).

See the [simtools container documentation](https://gammasim.github.io/simtools/user-guide/docker_files.html) for further details.

## Available Containers

### Standard Containers (with Native LightEmission Support)

- **`simtools-dev`**: Development environment with optional native C++ bindings
- **`simtools`**: Production environment with native bindings pre-compiled

Both containers now include native LightEmission support for improved performance.
The native bindings provide direct access to sim_telarray LightEmission C++ code.

## Development Usage

### Regular Development
```bash
# Standard development with subprocess fallback
podman run --rm -it -v "$(pwd)/:/workdir/external" ghcr.io/gammasim/simtools-dev:latest bash -c "source /workdir/env/bin/activate && cd /workdir/external/simtools && pip install -e . && bash"
```

### Development with Native LightEmission
Native bindings are auto-validated (and built if missing) when the container starts.
To disable auto-build:
```bash
podman run --rm -e SIMTOOLS_NO_AUTO_NATIVE=1 -it -v "$(pwd)/:/workdir/external" ghcr.io/gammasim/simtools-dev:latest bash
```

## Production Usage

### Applications (with automatic native support)
At entrypoint, the container checks the `_le` module for multi-event `ff_1m` support. If absent it builds it via CMake.
```bash
# Run flasher with native backend (auto-build if needed)
podman run --rm -v "$(pwd):/data" ghcr.io/gammasim/simtools:latest \
	simtools-simulate-flasher-native --telescope MSTS-04 --site South --light_source MSFx-FlashCam

# Disable auto-build and rely on already-present module (or fail)
podman run --rm -e SIMTOOLS_NO_AUTO_NATIVE=1 -v "$(pwd):/data" ghcr.io/gammasim/simtools:latest \
	simtools-simulate-flasher-native --telescope MSTS-04 --site South --light_source MSFx-FlashCam

# Force subprocess fallback (diagnostics)
podman run --rm -v "$(pwd):/data" ghcr.io/gammasim/simtools:latest \
	simtools-simulate-flasher-native --force_subprocess --telescope MSTS-04 --site South --light_source MSFx-FlashCam
```

## Database Connectivity

In case a local database is used:

```bash
podman run --rm -it -v "$(pwd)/:/workdir/external" --network simtools-mongo-network ghcr.io/gammasim/simtools-dev:latest bash -c "source /workdir/env/bin/activate && cd /workdir/external/simtools && pip install -e . && bash"
```

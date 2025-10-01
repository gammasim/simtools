# Containers

Container files are available for [simtools](https://github.com/gammasim/simtools) for both development and applications. Pre-built container images are available through the [simtools package registry](https://github.com/orgs/gammasim/packages?repo_name=simtools).

See the [simtools container documentation](https://gammasim.github.io/simtools/user-guide/docker_files.html) for further details.

## Available Containers

### Standard Containers (with Native LightEmission Support)

- **`simtools-dev`**: Development environment with optional native C++ bindings
- **`simtools`**: Production environment with native bindings pre-compiled

Both containers now include native LightEmission support for improved performance.
The native bindings provide direct access to sim_telarray LightEmission C++ code,
eliminating subprocess overhead for 2-5x performance improvement in flasher simulations.

## Development Usage

### Regular Development
```bash
# Standard development with subprocess fallback
podman run --rm -it -v "$(pwd)/:/workdir/external" ghcr.io/gammasim/simtools-dev:latest bash -c "source /workdir/env/bin/activate && cd /workdir/external/simtools && pip install -e . && bash"
```

### Development with Native LightEmission
```bash
# Build native bindings in container for performance
podman run --rm -it -v "$(pwd)/:/workdir/external" ghcr.io/gammasim/simtools-dev:latest bash -c "source /workdir/env/bin/activate && cd /workdir/external/simtools && build_native_in_container.sh && bash"
```

## Production Usage

### Applications (with automatic native support)
```bash
# Production containers attempt native build automatically, fall back to subprocess if needed
podman run --rm -v "$(pwd):/data" ghcr.io/gammasim/simtools:latest simtools-simulate-flasher --telescope MSTS-04 --site South --light_source MSFx-FlashCam

# Use new native-optimized application
podman run --rm -v "$(pwd):/data" ghcr.io/gammasim/simtools:latest simtools-simulate-flasher-native --telescope MSTS-04 --site South --light_source MSFx-FlashCam

# Force subprocess fallback for comparison/debugging
podman run --rm -v "$(pwd):/data" ghcr.io/gammasim/simtools:latest simtools-simulate-flasher-native --force_subprocess --telescope MSTS-04 --site South --light_source MSFx-FlashCam
```

## Database Connectivity

In case a local database is used:

```bash
podman run --rm -it -v "$(pwd)/:/workdir/external" --network simtools-mongo-network ghcr.io/gammasim/simtools-dev:latest bash -c "source /workdir/env/bin/activate && cd /workdir/external/simtools && pip install -e . && bash"
```

## Native Backend Benefits

- **Performance**: 2-5x faster than subprocess calls
- **Memory**: Lower memory overhead
- **Compatibility**: Graceful fallback if native build fails
- **Testing**: Easy comparison between native and subprocess modes
- **Simplicity**: Integrated into existing containers, no separate images needed

#!/bin/bash
# Container-optimized build script for native LightEmission support

set -e

echo "Building simtools with native LightEmission support in container..."

# Verify we're in a container with sim_telarray
if [ ! -d "/workdir/sim_telarray/LightEmission" ]; then
    echo "Error: sim_telarray LightEmission not found at /workdir/sim_telarray/LightEmission"
    echo "This script must run in a container based on corsika-simtel image"
    exit 1
fi

# Verify we're in the simtools directory
if [ ! -f "pyproject.toml" ] || [ ! -f "setup_extensions.py" ]; then
    echo "Error: Must run from simtools source directory"
    echo "Current directory: $(pwd)"
    echo "Expected files: pyproject.toml, setup_extensions.py"
    exit 1
fi

# Set up environment
export SIMTOOLS_BUILD_LE=1
export SIMTEL_PATH="/workdir/sim_telarray"

# Check if pybind11 is available
if ! python -c "import pybind11" 2>/dev/null; then
    echo "Installing pybind11..."
    pip install pybind11
fi

# Build extension
echo "Building native extension..."
echo "SIMTEL_PATH: $SIMTEL_PATH"
echo "LightEmission sources: $(ls -la $SIMTEL_PATH/LightEmission/)"

python setup_extensions.py build_ext --inplace

# Verify build
echo "Verifying build..."
if python -c "from simtools.light_emission.native_backend import HAS_NATIVE; print(f'Native: {HAS_NATIVE}'); exit(0 if HAS_NATIVE else 1)"; then
    echo "✓ Native LightEmission backend built successfully!"
else
    echo "✗ Native backend build failed"
    exit 1
fi

# Install in development mode if not already installed
if ! python -c "import simtools" 2>/dev/null; then
    echo "Installing simtools in development mode..."
    pip install -e .
fi

echo "Native build complete!"
echo ""
echo "Test with:"
echo "python -c \"from simtools.light_emission.native_backend import HAS_NATIVE; print(f'Native: {HAS_NATIVE}')\""
echo "simtools-simulate-flasher-native --help"

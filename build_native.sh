#!/bin/bash
# Build script for simtools LightEmission native backend
# Updated to work with or without LightEmission sources

set -e

echo "Building simtools with native LightEmission support..."

# Parse command line arguments
FORCE_BUILD=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_BUILD=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--force] [--verbose]"
            echo "  --force    Build even without LightEmission sources (placeholder mode)"
            echo "  --verbose  Enable verbose output"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check for LightEmission sources (optional)
if [ -z "$SIMTEL_PATH" ]; then
    echo "Warning: SIMTEL_PATH environment variable not set"
    if [ "$FORCE_BUILD" != "true" ]; then
        echo "Use --force to build with placeholder implementation, or set SIMTEL_PATH"
        exit 1
    fi
    echo "Proceeding with placeholder build..."
elif [ ! -d "$SIMTEL_PATH/LightEmission" ]; then
    echo "Warning: LightEmission directory not found at $SIMTEL_PATH/LightEmission"
    if [ "$FORCE_BUILD" != "true" ]; then
        echo "Use --force to build with placeholder implementation"
        exit 1
    fi
    echo "Proceeding with placeholder build..."
else
    echo "✓ Found LightEmission sources at: $SIMTEL_PATH/LightEmission"
fi

# Set build environment
export SIMTOOLS_BUILD_LE=1

# Install dependencies
echo "Installing pybind11 if needed..."
pip install pybind11 > /dev/null 2>&1 || pip install pybind11

# Build extension using new build script
echo "Building extension..."
if [ "$VERBOSE" = "true" ]; then
    python src/simtools/build_native_extension.py --verbose
else
    python src/simtools/build_native_extension.py
fi

# Install in development mode
echo "Installing simtools..."
if [ "$VERBOSE" = "true" ]; then
    pip install -e .
else
    pip install -e . > /dev/null 2>&1 || pip install -e .
fi

echo "Build complete!"
echo ""
echo "Test the installation:"
echo "python -c \"from simtools.light_emission.native_backend import HAS_NATIVE; print(f'Native: {HAS_NATIVE}')\""

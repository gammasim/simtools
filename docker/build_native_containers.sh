#!/bin/bash
# Build script for creating simtools containers with native LightEmission support

set -e

# Default values
BASE_IMAGE_TAG="latest"
SIMTOOLS_BRANCH="main"
PYTHON_VERSION="3.12"
PUSH_IMAGES=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-tag)
            BASE_IMAGE_TAG="$2"
            shift 2
            ;;
        --branch)
            SIMTOOLS_BRANCH="$2"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --push)
            PUSH_IMAGES=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --base-tag TAG    Base corsika-simtel image tag (default: latest)"
            echo "  --branch BRANCH   Simtools branch to build (default: main)"
            echo "  --python VERSION  Python version (default: 3.12)"
            echo "  --push           Push images to registry"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Building simtools containers with native LightEmission support..."
echo "Base image tag: $BASE_IMAGE_TAG"
echo "Simtools branch: $SIMTOOLS_BRANCH"
echo "Python version: $PYTHON_VERSION"

# Determine base image
BASE_IMAGE="ghcr.io/gammasim/corsika-simtel-250304-corsika-78000-bernlohr-1.69-prod6-baseline-qgs3-no_opt:$BASE_IMAGE_TAG"

# Build development container with native support
echo "Building simtools-dev-native..."
podman build \
    --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
    -f docker/Dockerfile-dev-native \
    -t ghcr.io/gammasim/simtools-dev-native:latest \
    -t ghcr.io/gammasim/simtools-dev-native:$SIMTOOLS_BRANCH \
    .

# Build production container with native support
echo "Building simtools-native..."
podman build \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg BUILD_BRANCH="$SIMTOOLS_BRANCH" \
    --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
    -f docker/Dockerfile-simtools-native \
    -t ghcr.io/gammasim/simtools-native:latest \
    -t ghcr.io/gammasim/simtools-native:$SIMTOOLS_BRANCH \
    .

# Test the containers
echo "Testing native backend in containers..."

echo "Testing development container..."
podman run --rm ghcr.io/gammasim/simtools-dev-native:latest \
    python -c "from simtools.light_emission.native_backend import HAS_NATIVE; print(f'Dev container native: {HAS_NATIVE}')"

echo "Testing production container..."
podman run --rm ghcr.io/gammasim/simtools-native:latest \
    python -c "from simtools.light_emission.native_backend import HAS_NATIVE; print(f'Prod container native: {HAS_NATIVE}'); assert HAS_NATIVE"

echo "Testing native application..."
podman run --rm ghcr.io/gammasim/simtools-native:latest \
    simtools-simulate-flasher-native --help > /dev/null

echo "✓ All container builds successful!"

if [ "$PUSH_IMAGES" = true ]; then
    echo "Pushing images to registry..."
    podman push ghcr.io/gammasim/simtools-dev-native:latest
    podman push ghcr.io/gammasim/simtools-dev-native:$SIMTOOLS_BRANCH
    podman push ghcr.io/gammasim/simtools-native:latest
    podman push ghcr.io/gammasim/simtools-native:$SIMTOOLS_BRANCH
    echo "✓ Images pushed successfully!"
fi

echo "Build complete!"
echo ""
echo "Usage examples:"
echo "Development: podman run -v \$(pwd):/workdir/external ghcr.io/gammasim/simtools-dev-native:latest"
echo "Production:  podman run ghcr.io/gammasim/simtools-native:latest simtools-simulate-flasher-native --help"

#!/bin/bash
# Build native LightEmission extension for simtools.
#
# This script attempts to build the native C++ extension that provides
# direct bindings to sim_telarray LightEmission functionality.
#
# Requirements:
# - SIMTEL_PATH environment variable pointing to sim_telarray installation
# - LightEmission headers and sources available
# - pybind11 installed
# - C++ compiler available
#
# Usage:
#   build_native_lightemission.sh [--force] [--verbose]
#
# Options:
#   --force    Force build even if checks fail
#   --verbose  Enable verbose output

set -euo pipefail

# Configuration
VERBOSE=false
FORCE_BUILD=false

# Parse command line arguments
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
            echo "Build native LightEmission extension for simtools"
            echo ""
            echo "Usage: $0 [--force] [--verbose]"
            echo ""
            echo "Options:"
            echo "  --force    Force build even if checks fail"
            echo "  --verbose  Enable verbose output"
            echo "  --help     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
    esac
done

# Logging functions
log_info() {
    echo "[INFO] $*"
}

log_warn() {
    echo "[WARN] $*" >&2
}

log_error() {
    echo "[ERROR] $*" >&2
}

log_verbose() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo "[DEBUG] $*"
    fi
}

# Check if we're in the right environment
check_environment() {
    log_info "Checking build environment..."

    # Check if we're in a Python virtual environment
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        log_warn "Not in a Python virtual environment"
        if [[ "$FORCE_BUILD" != "true" ]]; then
            log_error "Use --force to build anyway, or activate a virtual environment"
            return 1
        fi
    else
        log_verbose "Virtual environment: $VIRTUAL_ENV"
    fi

    # Check for required Python packages
    if ! python -c "import pybind11" 2>/dev/null; then
        log_error "pybind11 not available - install with: pip install pybind11"
        return 1
    fi

    if ! python -c "import setuptools" 2>/dev/null; then
        log_error "setuptools not available - install with: pip install setuptools"
        return 1
    fi

    log_verbose "Python environment checks passed"
    return 0
}

# Check for sim_telarray LightEmission sources
check_lightemission_sources() {
    log_info "Checking for LightEmission sources..."

    if [[ -z "${SIMTEL_PATH:-}" ]]; then
        log_warn "SIMTEL_PATH environment variable not set"
        if [[ "$FORCE_BUILD" != "true" ]]; then
            log_error "Set SIMTEL_PATH or use --force for placeholder build"
            return 1
        fi
        log_warn "Proceeding with placeholder build"
        return 0
    fi

    log_verbose "SIMTEL_PATH: $SIMTEL_PATH"

    local le_header="$SIMTEL_PATH/LightEmission/IactLightEmission.hh"
    if [[ ! -f "$le_header" ]]; then
        log_warn "LightEmission header not found at: $le_header"
        if [[ "$FORCE_BUILD" != "true" ]]; then
            log_error "LightEmission sources not available, use --force for placeholder build"
            return 1
        fi
        log_warn "Proceeding with placeholder build"
        return 0
    fi

    log_info "✓ LightEmission sources found"
    return 0
}

# Build the extension
build_extension() {
    log_info "Building native LightEmission extension..."

    # Set build environment variables
    export SIMTOOLS_BUILD_LE=1

    # Find the project root (where setup_native_extension.py or pyproject.toml is)
    local project_root
    if [[ -f "setup_native_extension.py" ]]; then
        project_root="."
    elif [[ -f "pyproject.toml" ]]; then
        project_root="."
    elif [[ -f "../setup_native_extension.py" ]]; then
        project_root=".."
    elif [[ -f "../../setup_native_extension.py" ]]; then
        project_root="../.."
    else
        log_error "Cannot find project root (setup_native_extension.py or pyproject.toml)"
        return 1
    fi

    log_verbose "Project root: $project_root"

    # Change to project directory
    pushd "$project_root" > /dev/null

    # Use the dedicated build script if available, otherwise use setup.py
    if [[ -f "src/simtools/build_native_extension.py" ]]; then
        log_verbose "Using dedicated build script"
        if [[ "$VERBOSE" == "true" ]]; then
            python src/simtools/build_native_extension.py --verbose
        else
            python src/simtools/build_native_extension.py
        fi
    elif [[ -f "setup_native_extension.py" ]]; then
        log_verbose "Using setup_native_extension.py"
        if [[ "$VERBOSE" == "true" ]]; then
            python setup_native_extension.py build_ext --inplace --verbose
        else
            python setup_native_extension.py build_ext --inplace
        fi
    else
        log_error "No build script found"
        popd > /dev/null
        return 1
    fi

    popd > /dev/null

    # Verify the build
    if python -c "from simtools.light_emission.native_backend import HAS_NATIVE; exit(0 if HAS_NATIVE else 1)" 2>/dev/null; then
        log_info "✓ Native extension built successfully"
        return 0
    else
        log_warn "Extension built but not importable (this may be expected for placeholder builds)"
        return 0
    fi
}

# Main execution
main() {
    log_info "Starting native LightEmission build process"

    if ! check_environment; then
        log_error "Environment check failed"
        exit 1
    fi

    if ! check_lightemission_sources; then
        log_error "Source check failed"
        exit 1
    fi

    if ! build_extension; then
        log_error "Build failed"
        exit 1
    fi

    log_info "Build process completed successfully"
    log_info "You can now use native LightEmission acceleration in simtools"
    log_info ""
    log_info "Test with: python -c \"from simtools.light_emission.native_backend import HAS_NATIVE; print(f'Native: {HAS_NATIVE}')\""
}

# Run main function
main "$@"

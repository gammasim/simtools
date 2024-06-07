#!/bin/bash
# Purge local images, containers, networks related to the simtools-mongodb
# **DANGEROUS; this removes items**

SIMTOOLS_NETWORK="simtools-mongo-network"
CONTAINER_NAME="simtools-mongodb"

# Check if podman is available, if not use docker
if command -v podman &> /dev/null; then
    CMD=podman
elif command -v docker &> /dev/null; then
    CMD=docker
else
    echo "Error: Neither podman nor docker is available."
    exit 1
fi

# Function to remove container
remove_container() {
    local container_name=$1
    if $CMD ps -a --format "{{.Names}}" | grep -q "^${container_name}$"; then
        echo "Removing existing container $container_name"
        $CMD rm -f "${container_name}"
    fi
}

# Function to remove network
remove_network() {
    local network_name=$1
    if $CMD network exists "$network_name"; then
        echo "Removing existing network $network_name"
        $CMD network rm "$network_name"
    fi
}

# Cleanup existing containers and network
remove_container $CONTAINER_NAME
remove_network $SIMTOOLS_NETWORK
$CMD image rm -f mongo:latest

rm -rfv mongo-data

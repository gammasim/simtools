#!/bin/bash
# Purge images, containers, networks related to a local simtools-db setup.
#
# **DANGEROUS; this script removes items**

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

# ask for confirmation, as this script removes items
read -r -p "Do you really want to continue? This script removes items. Type 'yes' to confirm: " user_input
if [ "$user_input" != "yes" ]; then
    echo "Operation aborted."
    exit 1
fi

# Remove container
remove_container() {
    local container_name=$1
    if $CMD ps -a --format "{{.Names}}" | grep -q "^${container_name}$"; then
        echo "Removing existing container $container_name"
        $CMD rm -f "${container_name}"
    fi
}

# Remove network
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

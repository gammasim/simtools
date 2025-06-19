#!/bin/bash
# Setup a local mongoDB database for the simtools projects.
#
# Assumes that podman or docker is installed and running

SIMTOOLS_NETWORK="simtools-mongo-network"
CONTAINER_NAME="simtools-mongodb"
HOST_NAME="local-simtools-mongodb"

# Check if podman is available, if not use docker
if command -v podman &> /dev/null; then
    CMD=podman
elif command -v docker &> /dev/null; then
    CMD=docker
else
    echo "Error: Neither podman nor docker is available."
    exit 1
fi

echo "Creating network $SIMTOOLS_NETWORK"
$CMD network create $SIMTOOLS_NETWORK

echo "Creating data directory ./mongo-data"
mkdir -p "$(pwd)"/mongo-data
chmod 755 "$(pwd)"/mongo-data

echo "Starting MongoDB container..."
$CMD run -d \
  --name $CONTAINER_NAME \
  --network $SIMTOOLS_NETWORK \
  --hostname $HOST_NAME \
  -e MONGO_INITDB_ROOT_USERNAME=root \
  -e MONGO_INITDB_ROOT_PASSWORD=example \
  -p 27017:27017 \
  -v "$(pwd)"/mongo-data:/data/db \
  mongo:8.0.10

echo "Waiting for MongoDB to be fully ready and root user to be available..."
RETRIES=30
until $CMD exec $CONTAINER_NAME mongosh admin -u root -p example --eval "db.runCommand({ connectionStatus: 1 })" >/dev/null 2>&1 || [ $RETRIES -eq 0 ]; do
  echo "Waiting for MongoDB to be ready and root authable... ($((RETRIES--)) retries left)"
  sleep 2
done
if [ $RETRIES -eq 0 ]; then
  echo "MongoDB did not start in time."
  $CMD logs $CONTAINER_NAME
  exit 1
fi
echo "MongoDB is ready and root authentication is working."

echo "Creating 'api' user..."
# This command *requires* the root user to be able to authenticate and have userAdminAnyDatabase role.
# If this fails, the root setup was indeed the problem.
if ! $CMD exec $CONTAINER_NAME mongosh admin -u root -p example --eval "
db.createUser({
  user: 'api',
  pwd: 'password',
  roles: [
    { role: 'readWriteAnyDatabase', db: 'admin' },
    { role: 'dbAdminAnyDatabase', db: 'admin' },
    { role: 'userAdminAnyDatabase', db: 'admin' },
  ]
});
"; then
  echo "Error: Failed to create 'api' user. Check root credentials or MongoDB logs."
  $CMD logs $CONTAINER_NAME
  exit 1
fi
echo "'api' user created successfully."

#!/bin/bash
# Upload a local dump of the remote mongoDB to a local mongoDB.
#
# Requires a dump directory with the database dump to be present in the current directory.
# Assumes that podman or docker is installed and running.

SIMTOOLS_NETWORK="simtools-mongo-network"
CONTAINER_NAME="simtools-mongodb"
SIMTOOLS_DB_SIMULATION_MODEL='CTAO-Simulation-Model-v0-3-0'  # Name of the database to be created

# Check if podman is available, if not use docker
if command -v podman &> /dev/null; then
    CMD=podman
elif command -v docker &> /dev/null; then
    CMD=docker
else
    echo "Error: Neither podman nor docker is available."
    exit 1
fi

echo "Create user for DB $SIMTOOLS_DB_SIMULATION_MODEL"
$CMD exec -it $CONTAINER_NAME mongosh admin -u root -p example --eval "
db.createUser({
  user: 'api',
  pwd: 'password',
  roles: [
    {
      role: 'readWrite',
      db: $SIMTOOLS_DB_SIMULATION_MODEL,
    },
  ]
});
"

echo "Upload existing dump of $SIMTOOLS_DB_SIMULATION_MODEL to DB."
$CMD run --rm \
  --network $SIMTOOLS_NETWORK \
  -v "$(pwd)"/dump:/dump \
  mongo:latest mongorestore \
  --host $CONTAINER_NAME \
  --port 27017 \
  -u api \
  -p password \
  --authenticationDatabase "admin" \
  -d "$SIMTOOLS_DB_SIMULATION_MODEL" \
  "/dump/$SIMTOOLS_DB_SIMULATION_MODEL"

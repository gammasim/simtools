#!/bin/bash
# Setup a local mongoDB database for the simtools projects
# Assumes that podman or docker is installed and running
# Requires a dump directory with the database dump to be present in the current directory
#
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

echo "Create network $SIMTOOLS_NETWORK"
$CMD network create $SIMTOOLS_NETWORK

echo "Data directory ./mongo-data"
mkdir -p "$(pwd)"/mongo-data
chmod 755 "$(pwd)"/mongo-data

# Start mongoDB
$CMD run -d \
  --name $CONTAINER_NAME \
  --network $SIMTOOLS_NETWORK \
  --hostname $HOST_NAME \
  -e MONGO_INITDB_ROOT_USERNAME=root \
  -e MONGO_INITDB_ROOT_PASSWORD=example \
  -p 27017:27017 \
  -v "$(pwd)"/mongo-data:/data/db \
  mongo:latest

echo "Waiting for MongoDB to start..."
# Loop until MongoDB is ready
RETRIES=30
until $CMD exec $CONTAINER_NAME mongosh --eval "db.runCommand({ ping: 1 })" >/dev/null 2>&1 || [ $RETRIES -eq 0 ]; do
  echo "Waiting for MongoDB to be ready... ($((RETRIES--)) retries left)"
  sleep 2
done
if [ $RETRIES -eq 0 ]; then
  echo "MongoDB did not start in time."
  exit 1
fi

# Verify root authentication
echo "Verifying root authentication..."
if ! $CMD exec $CONTAINER_NAME mongosh admin -u root -p example --eval "db.runCommand({ connectionStatus: 1 })"; then
  echo "Error: Root authentication failed."
  $CMD logs $CONTAINER_NAME
  exit 1
fi

# Create user
echo "Create user"
$CMD exec -it $CONTAINER_NAME mongosh admin -u root -p example --eval "
db.createUser({
  user: 'api',
  pwd: 'password',
  roles: [
    {
      role: 'readWrite',
      db: 'Staging-CTA-Simulation-Model-v0-3-0'
    },
  ]
});
"

echo "Upload existing DB dump to DB."
for db in $(ls dump); do
  echo "Uploading $db"
  $CMD run --rm \
    --network $SIMTOOLS_NETWORK \
    -v "$(pwd)"/dump:/dump \
    mongo:latest mongorestore \
    --host $CONTAINER_NAME \
    --port 27017 \
    -u api \
    -p password \
    --authenticationDatabase "admin" \
    -d "$db" \
    "/dump/$db"
done

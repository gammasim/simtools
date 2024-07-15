#!/bin/bash
# Upload model parameter from repository a local mongoDB.
#
# Assumes that podman or docker is installed and running.
# Requires configuration of local DB in .env file

CONTAINER_NAME="simtools-mongodb"
SIMTOOLS_DB_SIMULATION_MODEL='CTAO-Simulation-Model-v0-3-0'  # Name of the database to be created
SIMTOOLS_DB_SIMULATION_MODEL_URL="https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/model_parameters.git"
SIMTOOLS_DB_SIMULATION_MODEL_BRANCH="main"

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
    { role: 'readWriteAnyDatabase', db: 'admin' },
    { role: 'dbAdminAnyDatabase', db: 'admin' },
    { role: 'userAdminAnyDatabase', db: 'admin' },
    { role: 'readWrite', db: '$SIMTOOLS_DB_SIMULATION_MODEL' }
  ]
});
"

echo "Cloning model parameters from $SIMTOOLS_DB_SIMULATION_MODEL_URL"
rm -rf ./tmp_model_parameters
git clone -b $SIMTOOLS_DB_SIMULATION_MODEL_BRANCH $SIMTOOLS_DB_SIMULATION_MODEL_URL ./tmp_model_parameters

CURRENTDIR=$(pwd)
cd ./tmp_model_parameters/ || exit

# setup environment
filename=".env"
cat << EOF > "$filename"
SIMTOOLS_DB_API_PORT=27017 #Port on the MongoDB server
SIMTOOLS_DB_SERVER='localhost'
SIMTOOLS_DB_API_USER='api' # username for MongoDB
SIMTOOLS_DB_API_PW='password' # Password for MongoDB
SIMTOOLS_DB_API_AUTHENTICATION_DATABASE='admin'
SIMTOOLS_DB_SIMULATION_MODEL='$SIMTOOLS_DB_SIMULATION_MODEL'
EOF

# upload files to DB
model_directory="./model_versions/"
for dir in "${model_directory}"*/; do
  model_version=$(basename "${dir}")
  if [ "$model_version" = "metadata" ]; then
    simtools-db-add-model-parameters-from-repository-to-db \
    --input_path "${dir}"/ \
    --db_name $SIMTOOLS_DB_SIMULATION_MODEL \
    --type "metadata"
  else
    simtools-db-add-model-parameters-from-repository-to-db \
    --model_version "${model-version}" \
    --input_path "${dir}"/verified_model \
    --db_name $SIMTOOLS_DB_SIMULATION_MODEL \
    --type "model_parameters"
  fi
done

cd "$CURRENTDIR" || exit

rm -rf ./tmp_model_parameters

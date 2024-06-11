#!/bin/bash
# Upload model parameter from repository a local mongoDB.
#
# Assumes that podman or docker is installed and running.
# Requires configuration of local DB in .env file

CONTAINER_NAME="simtools-mongodb"
SIMTOOLS_DB_SIMULATION_MODEL='Staging-CTA-Simulation-Model-v0-3-0'
SIMTOOLS_DB_SIMULATION_MODEL_URL="https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/model_parameters.git"

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
git clone $SIMTOOLS_DB_SIMULATION_MODEL_URL ./tmp_model_parameters

model_directory="./tmp_model_parameters/model_versions/"

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

rm -rf ./tmp_model_parameters

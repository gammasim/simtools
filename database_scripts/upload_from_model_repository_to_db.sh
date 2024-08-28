#!/bin/bash
# Upload model parameter from repository to a local or remote mongoDB.
#

SIMTOOLS_DB_SIMULATION_MODEL_URL="https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/model_parameters.git"
SIMTOOLS_DB_SIMULATION_MODEL_BRANCH="main"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <DB simulation model name>"
    echo ""
    echo "Uses the .env file in the simtools directory for database configuration."
    exit 1
fi

SIMTOOLS_DB_SIMULATION_MODEL="$1"

echo "Cloning model parameters from $SIMTOOLS_DB_SIMULATION_MODEL_URL"
rm -rf ./tmp_model_parameters
git clone --depth=1 -b $SIMTOOLS_DB_SIMULATION_MODEL_BRANCH $SIMTOOLS_DB_SIMULATION_MODEL_URL ./tmp_model_parameters

CURRENTDIR=$(pwd)
cd ./tmp_model_parameters/ || exit
cp -f "$CURRENTDIR"/../.env .env

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
    --input_path "${dir}" \
    --db_name $SIMTOOLS_DB_SIMULATION_MODEL \
    --type "model_parameters"
  fi
done

cd "$CURRENTDIR" || exit

rm -rf ./tmp_model_parameters

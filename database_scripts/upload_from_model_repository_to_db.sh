#!/bin/bash
# Upload model parameter from repository to a local or remote mongoDB.
#
# Cover 'source .env': the script ensure that this file exists:
# shellcheck disable=SC1091

DB_SIMULATION_MODEL_URL="https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/model_parameters.git"
DB_SIMULATION_MODEL_BRANCH="main"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <DB simulation model name>"
    echo ""
    echo "Uses the .env file in the simtools directory for database configuration."
    exit 1
fi

DB_SIMULATION_MODEL="$1"

echo "Cloning model parameters from $DB_SIMULATION_MODEL_URL"
rm -rf ./tmp_model_parameters
git clone --depth=1 -b $DB_SIMULATION_MODEL_BRANCH $DB_SIMULATION_MODEL_URL ./tmp_model_parameters

CURRENTDIR=$(pwd)
cd ./tmp_model_parameters/ || exit
cp -f "$CURRENTDIR"/../.env .env

# ask for confirmation before uploading to remote DB
source .env
regex='^([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$'
echo "DB_SERVER: $SIMTOOLS_DB_SERVER"
if [[ $SIMTOOLS_DB_SERVER =~ $regex ]]; then
  read -r -p "Do you really want to upload to remote DB $SIMTOOLS_DB_SERVER? Type 'yes' to confirm: " user_input
  if [ "$user_input" != "yes" ]; then
      echo "Operation aborted."
      exit 1
  fi
fi

# upload files to DB
model_directory="./model_versions/"
for dir in "${model_directory}"*/; do
  simtools-db-add-model-parameters-from-repository-to-db \
  --model_version "$(basename "${dir}")" \
  --input_path "${dir}" \
  --db_name "$DB_SIMULATION_MODEL" \
  --type "model_parameters"
done

cd "$CURRENTDIR" || exit

rm -rf ./tmp_model_parameters

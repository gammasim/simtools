#!/bin/bash
# Upload model parameter from repository to a local or remote mongoDB.
#
# Execute this scripts from the ./database_scripts directory.
# Cover 'source .env': the script ensure that this file exists:
# shellcheck disable=SC1091

DB_SIMULATION_MODEL_URL="https://gitlab.cta-observatory.org/cta-science/simulations/simulation-model/simulation-models.git"
DB_SIMULATION_MODEL_BRANCH="main"

# Check that this script is not sourced but executed
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "This script must be executed, not sourced."
    echo "Usage: ./upload_from_model_repository_to_db.sh <DB simulation model name>"
    return 1
fi

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

CURRENT_DIR=$(pwd)
cd ./tmp_model_parameters/ || exit
cp -f "$CURRENT_DIR"/../.env .env

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

# upload model parameters to DB
model_directory="./simulation-models/model_parameters/"
simtools-db-add-simulation-model-from-repository-to-db \
  --input_path "${model_directory}" \
  --db_name "$DB_SIMULATION_MODEL" \
  --type "model_parameters"

# upload production tables to DB
production_directory="./simulation-models/productions"
simtools-db-add-simulation-model-from-repository-to-db \
  --input_path "${production_directory}" \
  --db_name "$DB_SIMULATION_MODEL" \
  --type "production_tables"

cd "$CURRENT_DIR" || exit

rm -rf ./tmp_model_parameters

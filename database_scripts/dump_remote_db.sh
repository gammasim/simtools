#!/bin/bash
# Dump a remote mongo DB to a local directory.
#
# Uses database definitions found in environment variables
# (from .env file defined in "../.env")

if [ -f ../.env ]; then
    while IFS='=' read -r key value || [[ -n "$key" ]]; do
        # Skip lines starting with '#' or lines containing only whitespace
        if [[ ! "$key" =~ ^\ *# && -n "$key" ]]; then
            export "$key"="${value%%\#*}"
        fi
    done < ../.env
fi


# Clean up environment variables
clean_var() {
    echo "${1//[[:space:]\']/}"
}

SIMTOOLS_DB_SERVER=$(clean_var "$SIMTOOLS_DB_SERVER")
SIMTOOLS_DB_API_PORT=$(clean_var "$SIMTOOLS_DB_API_PORT")
SIMTOOLS_DB_API_AUTHENTICATION_DATABASE=$(clean_var "$SIMTOOLS_DB_API_AUTHENTICATION_DATABASE")
SIMTOOLS_DB_API_USER=$(clean_var "$SIMTOOLS_DB_API_USER")
SIMTOOLS_DB_API_PW=$(clean_var "$SIMTOOLS_DB_API_PW")
SIMTOOLS_DB_SIMULATION_MODEL=$(clean_var "$SIMTOOLS_DB_SIMULATION_MODEL")

[[ "$1" ]] && DB_TO_DUMP="$1" || DB_TO_DUMP="$SIMTOOLS_DB_SIMULATION_MODEL"

echo "Server: $SIMTOOLS_DB_SERVER"
echo "Port: $SIMTOOLS_DB_API_PORT"
echo "Auth DB: $SIMTOOLS_DB_API_AUTHENTICATION_DATABASE"
echo "User: $SIMTOOLS_DB_API_USER"
echo "Database to dump: $DB_TO_DUMP"

mkdir -p dump

if ! mongodump --uri="mongodb://${SIMTOOLS_DB_SERVER}:${SIMTOOLS_DB_API_PORT}" \
     --ssl --tlsInsecure \
     --username="$SIMTOOLS_DB_API_USER" --password="$SIMTOOLS_DB_API_PW" \
     --authenticationDatabase="$SIMTOOLS_DB_API_AUTHENTICATION_DATABASE" \
     --db="$DB_TO_DUMP" --out="./dump/"; then

    echo "Failed to connect to MongoDB server. Please check connection settings and try again."
    exit 1
fi

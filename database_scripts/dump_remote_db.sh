#!/bin/bash
# Dump a remove mongo DB to a local directory
# uses environment variables from .env file defined in "../.env"

if [ -f ../.env ]; then
    while IFS='=' read -r key value || [[ -n "$key" ]]; do
        # Skip lines starting with '#' or lines containing only whitespace
        if [[ ! "$key" =~ ^\ *# && -n "$key" ]]; then
            export "$key"="${value%%\#*}"
        fi
    done < ../.env
fi

echo "Dumping simulation model database $SIMTOOLS_DB_SIMULATION_MODEL"
mkdir -p dump

mongodump --uri="mongodb://${SIMTOOLS_DB_SERVER}:${SIMTOOLS_DB_API_PORT}" \
 --ssl --tlsInsecure \
 --username="$SIMTOOLS_DB_USER" --password="$SIMTOOLS_DB_API_PW" \
 --authenticationDatabase="$SIMTOOLS_DB_API_AUTHENTICATION_DATABASE" \
 --db="$DB_TO_COPY" --out="./dump/"

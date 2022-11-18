#!/usr/bin/bash
export DB_API_PORT=27017 #Port on the MongoDB server
export DB_SERVER='cta-simpipe-protodb.zeuthen.desy.de' # MongoDB server
export DB_API_USER=YOUR_USERNAME # username for MongoDB: ask the responsible person
export DB_API_PW=YOUR_PASSWORD # Password for MongoDB: ask the responsible person
export DB_API_AUTHENTICATION_DATABASE='admin'
export SIMTEL_PATH='/workdir/sim_telarray'

# The dashboards to monitor the MongoDB instance are in (accessible only from within DESY)
# https://statspub.zeuthen.desy.de/d/4vXnWwMGz/mongodb?orgId=1&refresh=30s
# https://statspub.zeuthen.desy.de/d/tBkrQGNmz/mongodb-wiredtiger?orgId=1&refresh=1m

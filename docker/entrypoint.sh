#!/bin/bash
# entrypoint script to configure development environment

cd /workdir/external/simtools
pip install -e .

if [[ -e "set_DB_environ.sh" ]]; then
    source ./set_DB_environ.sh
fi

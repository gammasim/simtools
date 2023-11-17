docker run --rm -it --env-file .env \
    -v "$(pwd):/workdir/external" \
    ghcr.io/gammasim/simtools-prod:latest \
    simtools-derive-mirror-rnda \
    --site North \
    --telescope MST-FlashCam-D \
    --containment_fraction 0.8 \
    --mirror_list /workdir/external/tests/resources/MLTdata-preproduction.ecsv \
    --psf_measurement /workdir/external/tests/resources/MLTdata-preproduction.ecsv \
    --rnda 0.0063 \
    --test \
    --output_path /workdir/external/

#!/usr/bin/python3

import json

from astropy import units as u
from astropy.table import Table

from simtools.constants import SCHEMA_PATH
from simtools.data_model import data_reader
from simtools.data_model.metadata_collector import MetadataCollector


def test_read_table_from_file_and_validate(tmp_test_directory, args_dict_site):
    table_file = tmp_test_directory / "mirror_measurement.ecsv"
    Table(
        {
            "mirror_panel_id": ["1"],
            "mirror_curvature_radius": [100.0] * u.cm,
            "psf": [1.0] * u.cm,
            "psf_opt": [0.8] * u.cm,
        },
    ).write(table_file, format="ascii.ecsv")
    schema_file = SCHEMA_PATH / "input/MST_mirror_2f_measurements.schema.yml"

    metadata = MetadataCollector(args_dict=args_dict_site, clean_meta=False).top_level_meta
    metadata["cta"]["instrument"].pop("ID", None)
    metadata["cta"]["instrument"]["id"] = "MSTS-07"
    metadata["cta"]["product"]["data"]["model"]["url"] = str(schema_file)
    metadata_file = tmp_test_directory / "mirror_measurement.json"
    metadata_file.write_text(json.dumps(metadata), encoding="utf-8")
    assert isinstance(
        data_reader.read_table_from_file(
            table_file,
            validate=True,
            schema_file=schema_file,
            metadata_file=metadata_file,
        ),
        Table,
    )
    assert isinstance(
        data_reader.read_table_from_file(
            table_file,
            validate=True,
            metadata_file=metadata_file,
        ),
        Table,
    )

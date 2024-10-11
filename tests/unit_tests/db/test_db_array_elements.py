#!/usr/bin/python3

import time

import pytest

from simtools.db import db_array_elements


def test_get_array_elements(db, model_version):

    time_1 = time.time()
    db_array_elements.get_array_elements(
        db.get_collection(db_name=None, collection_name="telescopes"),
        db.model_version(model_version),
    )
    time_2 = time.time()
    available_telescopes = db_array_elements.get_array_elements(
        db.get_collection(db_name=None, collection_name="telescopes"),
        db.model_version(model_version),
    )
    time_3 = time.time()

    # check that the second call is much faster than the first one
    assert (time_2 - time_1) > 0.1 * (time_3 - time_2)

    expected_telescope_names = {
        "LSTN-01": "LSTN-design",
        "LSTN-02": "LSTN-design",
        "LSTN-03": "LSTN-design",
        "LSTN-04": "LSTN-design",
        "MSTN-15": "MSTN-design",
        "MSTS-10": "MSTS-design",
        "MSTS-301": "MSTN-design",
    }
    for _t in expected_telescope_names:
        assert _t in available_telescopes
        assert expected_telescope_names[_t] in available_telescopes[_t]

    available_calibration_devices = db_array_elements.get_array_elements(
        db.get_collection(db_name=None, collection_name="calibration_devices"),
        db.model_version(model_version),
    )
    expected_calibration_devices = {
        "ILLN-01": "ILLN-design",
        "ILLS-02": "ILLS-design",
    }
    for _d in expected_calibration_devices:
        assert _d in available_calibration_devices
        assert expected_calibration_devices[_d] in available_calibration_devices[_d]

    with pytest.raises(ValueError, match=r"^No array elements found in DB collection"):
        db_array_elements.get_array_elements(
            db.get_collection(db_name=None, collection_name="wrong_collection"),
            db.model_version(model_version),
        )


def test_get_array_element_list_for_db_query(db, model_version):

    assert db_array_elements.get_array_element_list_for_db_query(
        "LSTN-01", db=db, model_version=model_version, collection="telescopes"
    ) == ["LSTN-design", "LSTN-01"]

    assert db_array_elements.get_array_element_list_for_db_query(
        "MSTS-10", db=db, model_version=model_version, collection="telescopes"
    ) == ["MSTS-design", "MSTS-10"]

    assert db_array_elements.get_array_element_list_for_db_query(
        "MSTS-301", db=db, model_version=model_version, collection="telescopes"
    ) == ["MSTN-design", "MSTS-301"]

    assert db_array_elements.get_array_element_list_for_db_query(
        "MSTS-design", db=db, model_version=model_version, collection="telescopes"
    ) == ["MSTS-design"]

    with pytest.raises(ValueError, match=r"^Array element MSTS-301 not found in DB."):
        db_array_elements.get_array_element_list_for_db_query(
            "MSTS-301", db=db, model_version=model_version, collection="calibration_devices"
        )

    assert db_array_elements.get_array_element_list_for_db_query(
        "LSTN-02", db=db, model_version=model_version, collection="configuration_sim_telarray"
    ) == ["LSTN-design"]


def test_get_array_elements_of_type(db, model_version):
    available_telescopes = db_array_elements.get_array_elements_of_type(
        array_element_type="LSTN", db=db, model_version=model_version, collection="telescopes"
    )
    assert available_telescopes == ["LSTN-01", "LSTN-02", "LSTN-03", "LSTN-04"]

    available_calibration_devices = db_array_elements.get_array_elements_of_type(
        array_element_type="ILLS",
        db=db,
        model_version=model_version,
        collection="calibration_devices",
    )
    assert available_calibration_devices == ["ILLS-01", "ILLS-02", "ILLS-03", "ILLS-04"]

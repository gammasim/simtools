#!/usr/bin/python3

import pytest
from astropy import units as u

import simtools.util.general as gen
from simtools.corsika.corsika_output import CorsikaOutput

test_file_name = "tel_output_10GeV-2-gamma-20deg-CTAO-South.dat"


@pytest.fixture
def corsika_output_file(io_handler):
    corsika_output_file = gen.find_file(
        test_file_name,
        io_handler.get_output_directory(dir_type="corsika_output", test=True),
    )
    return corsika_output_file


@pytest.fixture
def corsika_output_instance(db, io_handler, corsika_output_file):
    db.export_file_db(
        db_name="test-data",
        dest=io_handler.get_output_directory(dir_type="corsika_output", test=True),
        file_name=test_file_name,
    )
    return CorsikaOutput(corsika_output_file)


def test_extract_information(corsika_output_instance):
    list_of_variables = [
        "wave_tot",
        "x_photon_positions",
        "y_photon_positions",
        "x_cos",
        "y_cos",
        "z_photon_emission",
        "time_since_first_interaction",
        "distance",
    ]
    list_of_units = [u.nm, u.m, u.m, None, None, u.m, u.ns, u.m]
    for step, variable in enumerate(list_of_variables):
        information = getattr(corsika_output_instance, variable)
        assert len(information) == 11634
        try:
            unit = information.unit
        except AttributeError:
            unit = None
        assert unit == list_of_units[step]

    assert pytest.approx(corsika_output_instance.num_photons_tot[0]) == 2543.348

#!/usr/bin/python3

import pytest

from simtools.model import model_utils


def test_is_two_mirror_telescope():

    assert not model_utils.is_two_mirror_telescope("LSTN-01")
    assert not model_utils.is_two_mirror_telescope("MSTN-01")
    assert not model_utils.is_two_mirror_telescope("MSTS-01")
    assert model_utils.is_two_mirror_telescope("SSTS-01")
    assert model_utils.is_two_mirror_telescope("SCTS-25")


def test_compute_telescope_transmission():

    pars = [0.8, 0, 0.0, 0.0, 0.0]
    off_axis = 0.0
    assert pytest.approx(model_utils.compute_telescope_transmission(pars, off_axis)) == pars[0]

    pars = [0.898, 1, 0.016, 4.136, 1.705, 0.0]
    off_axis = 0.0
    assert pytest.approx(model_utils.compute_telescope_transmission(pars, off_axis)) == pars[0]

    pars = [0.898, 1, 0.016, 4.136, 1.705, 0.0]
    off_axis = 2.0
    assert pytest.approx(model_utils.compute_telescope_transmission(pars, off_axis)) == 0.8938578

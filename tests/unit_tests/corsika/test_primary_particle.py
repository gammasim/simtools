#!/usr/bin/python3

import logging

import pytest
from particle import Corsika7ID

from simtools.corsika.primary_particle import PrimaryParticle

logger = logging.getLogger()


def test_init():
    p = PrimaryParticle()

    assert p.corsika7_id is None

    with pytest.raises(
        ValueError, match=r"Both 'particle_id_type' and 'particle_id' must be provided together\."
    ):
        PrimaryParticle(particle_id_type="corsika7_id")
    with pytest.raises(
        ValueError, match=r"Both 'particle_id_type' and 'particle_id' must be provided together\."
    ):
        PrimaryParticle(particle_id_type="common_name")
    with pytest.raises(ValueError, match="Particle ID type must be one of"):
        PrimaryParticle(particle_id_type="invalid_type", particle_id=14)


def test_str():
    p = PrimaryParticle(particle_id_type="corsika7_id", particle_id=14)
    assert str(p) == "proton (14, 2212, p)"

    fe = PrimaryParticle(particle_id_type="corsika7_id", particle_id=5626)
    assert str(fe) == "iron (5626, 1000260560, Fe56)"


def test_corsika7_id():
    p = PrimaryParticle(particle_id_type="corsika7_id", particle_id=14)

    assert p.corsika7_id == 14
    assert p.name == "proton"
    assert p.pdg_id == 2212

    si = PrimaryParticle(particle_id_type="corsika7_id", particle_id=2814)
    assert si.corsika7_id == 2814
    assert si.name == "silicon"
    assert si.pdg_id == 1000140280

    with pytest.raises(ValueError, match="Invalid CORSIKA7 ID: 9999"):
        PrimaryParticle(particle_id_type="corsika7_id", particle_id=9999)

    eta = PrimaryParticle(particle_id_type="corsika7_id", particle_id=17)
    assert eta.corsika7_id == 17
    assert eta.name == "eta"
    assert eta.pdg_id == 221


def test_eventio_id():
    test_dict = {
        0: "gamma",
        1: "electron",
        -1: "positron",
        2: "muon-",
        -2: "muon+",
        100: "neutron",
        101: "proton",
        -101: "p~",
        402: "helium",
        1206: "C12",
        1407: "nitrogen",
        2814: "silicon",
        5626: "iron",
    }
    for eventio_id, name in test_dict.items():
        p = PrimaryParticle(particle_id_type="eventio_id", particle_id=eventio_id)
        assert p.eventio_id == eventio_id
        assert p.name == name

    with pytest.raises(ValueError, match="Invalid EventIO ID: 9999"):
        PrimaryParticle(particle_id_type="eventio_id", particle_id=9999)


def test_common_name():
    p = PrimaryParticle(particle_id_type="common_name", particle_id="proton")

    assert p.corsika7_id == 14
    assert p.name == "proton"
    assert p.pdg_id == 2212

    pi0 = PrimaryParticle(particle_id_type="common_name", particle_id="pi0")
    assert pi0.corsika7_id == 7
    assert pi0.name == "pi0"
    assert pi0.pdg_id == 111

    with pytest.raises(ValueError, match=r"Found more than one particle with name pi"):
        PrimaryParticle(particle_id_type="common_name", particle_id="pi")

    with pytest.raises(
        ValueError, match="Invalid particle name: the_particle_which_explains_nothing"
    ):
        PrimaryParticle(
            particle_id_type="common_name", particle_id="the_particle_which_explains_nothing"
        )


def test_pdg_id(caplog):
    p = PrimaryParticle(particle_id_type="pdg_id", particle_id=2212)

    assert p.corsika7_id == 14
    assert p.name == "proton"
    assert p.pdg_id == 2212

    fe = PrimaryParticle(particle_id_type="pdg_id", particle_id=1000260560)
    assert fe.corsika7_id == 5626
    assert fe.name == "iron"
    assert fe.pdg_id == 1000260560

    pi0 = PrimaryParticle(particle_id_type="pdg_id", particle_id=111)
    assert pi0.corsika7_id == 7
    assert pi0.name == "pi0"

    with pytest.raises(ValueError, match="Invalid DPG ID: 9999"):
        PrimaryParticle(particle_id_type="pdg_id", particle_id=9999)


def test_particle_names():
    for _, ids in PrimaryParticle.particle_names().items():
        assert Corsika7ID(ids["corsika7_id"]).to_pdgid().numerator

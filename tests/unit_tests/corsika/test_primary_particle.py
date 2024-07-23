#!/usr/bin/python3

import logging

import pytest
from particle import Corsika7ID

from simtools.corsika.primary_particle import PrimaryParticle

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_init():
    p = PrimaryParticle()

    assert p.corsika7_id is None


def test_str():
    p = PrimaryParticle(corsika7_id=14)
    assert str(p) == "proton (14, 2212, p)"

    fe = PrimaryParticle(corsika7_id=5626)
    assert str(fe) == "iron (5626, 1000260560, Fe56)"


def test_corsika7_id():
    p = PrimaryParticle(corsika7_id=14)

    assert p.corsika7_id == 14
    assert p.name == "proton"
    assert p.pdg_id == 2212

    si = PrimaryParticle(corsika7_id=2814)
    assert si.corsika7_id == 2814
    assert si.name == "silicon"
    assert si.pdg_id == 1000140280

    with pytest.raises(ValueError, match="Invalid CORSIKA7 ID: 9999"):
        PrimaryParticle(corsika7_id=9999)


def test_common_name():
    p = PrimaryParticle(name="proton")

    assert p.corsika7_id == 14
    assert p.name == "proton"
    assert p.pdg_id == 2212

    pi0 = PrimaryParticle(name="pi0")
    assert pi0.corsika7_id == 7
    assert pi0.name == "pi0"
    assert pi0.pdg_id == 111

    with pytest.raises(ValueError, match=r"Found more than one particle with name pi"):
        PrimaryParticle(name="pi")

    with pytest.raises(
        ValueError, match="Invalid particle name: the_particle_which_explains_nothing"
    ):
        PrimaryParticle(name="the_particle_which_explains_nothing")


def test_pdg_id(caplog):
    p = PrimaryParticle(pdg_id=2212)

    assert p.corsika7_id == 14
    assert p.name == "proton"
    assert p.pdg_id == 2212

    fe = PrimaryParticle(pdg_id=1000260560)
    assert fe.corsika7_id == 5626
    assert fe.name == "iron"
    assert fe.pdg_id == 1000260560

    pi0 = PrimaryParticle(pdg_id=111)
    assert pi0.corsika7_id == 7
    assert pi0.name == "pi0"

    with pytest.raises(ValueError, match="Invalid DPG ID: 9999"):
        PrimaryParticle(pdg_id=9999)


def test_particle_names():
    for _, ids in PrimaryParticle.particle_names().items():
        assert Corsika7ID(ids["corsika7_id"]).to_pdgid().numerator

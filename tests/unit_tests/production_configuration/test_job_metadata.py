"""Tests for simulation job metadata generation."""

from types import SimpleNamespace

import astropy.units as u
import pytest

from simtools.production_configuration.job_metadata import build_simulation_job_metadata


def _args(**updates):
    args = {
        "array_layout_name": "CTAO-South-Alpha",
        "site": "South",
        "primary": "Gamma",
        "azimuth_angle": 190 * u.deg,
        "zenith_angle": 20 * u.deg,
        "view_cone": (0 * u.deg, 1.5 * u.deg),
        "model_version": "7.0.0",
    }
    args.update(updates)
    return args


def _simulator(*array_elements, run_number=12):
    return SimpleNamespace(
        array_models=[SimpleNamespace(array_elements=dict.fromkeys(array_elements))],
        run_number=run_number,
    )


def test_build_simulation_job_metadata_uses_catalog_conventions():
    metadata = build_simulation_job_metadata(
        _args(dec=-45 * u.deg, ha=123 * u.deg),
        _simulator("MSTS-01", "SCTS-01"),
    )

    assert metadata == {
        "array_layout": "CTAO-South-Alpha",
        "site": "Paranal",
        "particle": "gamma",
        "phiP": 10.0,
        "thetaP": 20.0,
        "sct": "True",
        "view_cone": "0.0_deg_1.5_deg",
        "runNumber": 12,
        "model_version": "7.0.0",
        "dec": -45.0,
        "ha": 123.0,
    }


def test_build_simulation_job_metadata_omits_missing_coordinates_and_sets_sct_false():
    metadata = build_simulation_job_metadata(
        _args(site="North", azimuth_angle=180 * u.deg),
        _simulator("LSTN-01", run_number=5),
    )

    assert metadata["site"] == "LaPalma"
    assert metadata["phiP"] == pytest.approx(0.0)
    assert metadata["sct"] == "False"
    assert metadata["runNumber"] == 5
    assert "dec" not in metadata
    assert "ha" not in metadata

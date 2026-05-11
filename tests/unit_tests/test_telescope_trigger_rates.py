from pathlib import Path
from unittest.mock import MagicMock, patch

import astropy.units as u
import pytest
import yaml
from ctao_cr_spectra.definitions import IRFDOC_ELECTRON_SPECTRUM, IRFDOC_PROTON_SPECTRUM
from ctao_cr_spectra.spectral import LogParabola, PowerLaw

from simtools.telescope_trigger_rates import (
    _integrate_energy_spectrum,
    _load_spectrum_from_file,
    get_cosmic_ray_spectrum,
    telescope_trigger_rates,
)

FILE_SIMTEL = "test_file.simtel"


def test_telescope_trigger_rates_with_array_layout_name():
    args_dict = {
        "array_layout_name": "test_layout",
        "site": "test_site",
        "model_version": "1.0.0",
        "event_data_file": FILE_SIMTEL,
        "plot_histograms": False,
    }

    with (
        patch(
            "simtools.telescope_trigger_rates.get_array_elements_from_db_for_layouts"
        ) as mock_get_array_elements,
        patch("simtools.telescope_trigger_rates.EventDataHistograms") as mock_histograms,
    ):
        mock_get_array_elements.return_value = {"array1": [1, 2, 3]}
        mock_histograms_instance = MagicMock()
        mock_histograms.return_value = mock_histograms_instance

        telescope_trigger_rates(args_dict)

        mock_get_array_elements.assert_called_once_with("test_layout", "test_site", "1.0.0")
        mock_histograms.assert_called_once_with(
            FILE_SIMTEL, array_name="array1", telescope_list=[1, 2, 3]
        )
        mock_histograms_instance.fill.assert_called_once()
        mock_histograms_instance.plot.assert_not_called()


def test_telescope_trigger_rates_without_array_layout_name():
    args_dict = {
        "telescope_ids": Path("test_telescope_ids.txt"),
        "event_data_file": FILE_SIMTEL,
        "plot_histograms": True,
    }

    with (
        patch(
            "simtools.telescope_trigger_rates.ascii_handler.collect_data_from_file"
        ) as mock_collect_data,
        patch("simtools.telescope_trigger_rates.EventDataHistograms") as mock_histograms,
        patch("simtools.telescope_trigger_rates.plot_simtel_event_histograms.plot") as mock_plot,
        patch("simtools.telescope_trigger_rates.io_handler.IOHandler") as mock_io_handler,
    ):
        mock_collect_data.return_value = {"telescope_configs": {"array1": [1, 2, 3]}}
        mock_histograms_instance = MagicMock()
        mock_histograms.return_value = mock_histograms_instance
        mock_io_handler_instance = MagicMock()
        mock_io_handler.return_value = mock_io_handler_instance
        mock_io_handler_instance.get_output_directory.return_value = Path("output_dir")

        telescope_trigger_rates(args_dict)

        mock_collect_data.assert_called_once_with(Path("test_telescope_ids.txt"))
        mock_histograms.assert_called_once_with(
            FILE_SIMTEL, array_name="array1", telescope_list=[1, 2, 3]
        )
        mock_histograms_instance.fill.assert_called_once()
        mock_plot.assert_called_once_with(
            mock_histograms_instance.histograms, output_path=Path("output_dir"), array_name="array1"
        )


def test_get_cosmic_ray_spectrum_default():
    """No arguments: should return the proton spectrum."""
    assert get_cosmic_ray_spectrum() is IRFDOC_PROTON_SPECTRUM


def test_get_cosmic_ray_spectrum_known_particle():
    assert get_cosmic_ray_spectrum(primary_particle="proton") is IRFDOC_PROTON_SPECTRUM
    assert get_cosmic_ray_spectrum(primary_particle="electron") is IRFDOC_ELECTRON_SPECTRUM


def test_get_cosmic_ray_spectrum_unknown_particle_falls_back(caplog):
    """Unknown primary particle should log a warning and return the proton spectrum."""
    with caplog.at_level("WARNING"):
        result = get_cosmic_ray_spectrum(primary_particle="gamma")
    assert result is IRFDOC_PROTON_SPECTRUM
    assert "gamma" in caplog.text


def test_get_cosmic_ray_spectrum_from_file(tmp_test_directory):
    """Spectrum loaded from YAML file takes priority over primary_particle."""
    spectrum_file = tmp_test_directory / "spectrum.yml"
    config = {
        "type": "PowerLaw",
        "normalization": 9.8e-6,
        "normalization_unit": "1 / (cm2 s TeV sr)",
        "index": -2.7,
    }
    spectrum_file.write_text(yaml.dump(config), encoding="utf-8")

    result = get_cosmic_ray_spectrum(primary_particle="proton", cr_spectrum_file=spectrum_file)
    assert isinstance(result, PowerLaw)
    assert result.index == pytest.approx(-2.7)


def test_load_spectrum_from_file_power_law(tmp_test_directory):
    spectrum_file = tmp_test_directory / "pl.yml"
    config = {
        "type": "PowerLaw",
        "normalization": 9.8e-6,
        "normalization_unit": "1 / (cm2 s TeV sr)",
        "index": -2.62,
        "e_ref": 1.0,
        "e_ref_unit": "TeV",
    }
    spectrum_file.write_text(yaml.dump(config), encoding="utf-8")

    spectrum = _load_spectrum_from_file(spectrum_file)
    assert isinstance(spectrum, PowerLaw)
    assert spectrum.index == pytest.approx(-2.62)
    assert spectrum.e_ref.value == pytest.approx(1.0)


def test_load_spectrum_from_file_log_parabola(tmp_test_directory):
    spectrum_file = tmp_test_directory / "lp.yml"
    config = {
        "type": "LogParabola",
        "normalization": 3.23e-11,
        "normalization_unit": "1 / (cm2 s TeV sr)",
        "a": -2.47,
        "b": -0.24,
    }
    spectrum_file.write_text(yaml.dump(config), encoding="utf-8")

    spectrum = _load_spectrum_from_file(spectrum_file)
    assert isinstance(spectrum, LogParabola)
    assert spectrum.a == pytest.approx(-2.47)
    assert spectrum.b == pytest.approx(-0.24)


def test_load_spectrum_from_file_unknown_type_raises(tmp_test_directory):
    spectrum_file = tmp_test_directory / "bad.yml"
    config = {
        "type": "BrokenPowerLaw",
        "normalization": 1.0,
        "normalization_unit": "1 / (cm2 s TeV sr)",
    }
    spectrum_file.write_text(yaml.dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported spectrum type"):
        _load_spectrum_from_file(spectrum_file)


def test_integrate_energy_spectrum_power_law():
    """PowerLaw uses analytical integration."""
    spectrum = PowerLaw(
        normalization=9.8e-6 / (u.cm**2 * u.s * u.TeV * u.sr),
        index=-2.62,
        e_ref=1.0 * u.TeV,
    )
    result = _integrate_energy_spectrum(spectrum, 1.0 * u.TeV, 2.0 * u.TeV)
    expected = spectrum.integrate_energy(1.0 * u.TeV, 2.0 * u.TeV)
    assert result.decompose().value == pytest.approx(expected.decompose().value, rel=1e-6)


def test_integrate_energy_spectrum_log_parabola():
    """LogParabola uses numerical integration; result is physically reasonable."""
    spectrum = LogParabola(
        normalization=3.23e-11 / (u.cm**2 * u.s * u.TeV * u.sr),
        a=-2.47,
        b=-0.24,
    )
    result = _integrate_energy_spectrum(spectrum, 1.0 * u.TeV, 2.0 * u.TeV)
    assert result.value > 0
    # Cross-check that units reduce to flux * energy (per cm2 s sr)
    assert result.unit.is_equivalent(u.cm**-2 * u.s**-1 * u.sr**-1)

#!/usr/bin/python3
# pylint: disable=protected-access

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import astropy.units as u
import pytest
from astropy.table import QTable

from simtools.ray_tracing import incident_angles as ia
from simtools.ray_tracing import parameter_changes
from simtools.ray_tracing.incident_angles import IncidentAnglesCalculator
from simtools.utils.general import cleanup_intermediate_files

_IMAGING_HEADER = """# Column 2: Pixel number hit
# Column 22: Angle to pixel normal (if pixel hit) or focal surface normal [deg.]
# Column 26: Angle of incidence at focal surface, w.r.t. optical axis [deg.]
# Column 32: Angle of incidence on primary mirror, w.r.t. normal [deg.]
# Column 36: Angle of incidence on secondary mirror, w.r.t. normal [deg.]
"""


@pytest.fixture
def config_data():
    return {
        "telescope": "LSTN-01",
        "site": "North",
        "model_version": "prod6",
        "off_axis_angle": 0.0 * u.deg,
        "source_distance": 10.0 * u.km,
        "number_of_photons": 1000,
        "debug_plots": True,
    }


@pytest.fixture
def mock_models(monkeypatch):
    tel = MagicMock()
    tel.name = "LST-1"
    tel.config_file_path = Path("cfg.cfg")
    tel.config_file_directory = Path()
    tel.configuration_writers = {"sim_telarray": MagicMock()}
    tel.get_parameter_value.side_effect = lambda key: {
        "focal_length": 280.0,
        "mirror_class": 2,
    }.get(key, 0.0)

    site = MagicMock()
    site.site = "North"
    site.get_parameter_value.side_effect = lambda key: (
        2150.0 if key == "corsika_observation_level" else 0.0
    )

    dummy = object()
    monkeypatch.setattr(ia, "initialize_simulation_models", lambda *a, **k: (tel, site, dummy))
    return SimpleNamespace(tel=tel, site=site)


@pytest.fixture
def calculator(mock_models, config_data, tmp_test_directory):
    return IncidentAnglesCalculator(
        config_data=config_data,
        output_dir=tmp_test_directory,
        label="test-label",
    )


def test_initialization(calculator, config_data):
    assert calculator.zenith_angle_deg == pytest.approx(0)
    assert calculator.config_data == config_data
    assert calculator.output_dir.is_dir()
    assert calculator.results is None
    # subdirectories are created
    assert calculator.logs_dir.is_dir()
    assert calculator.scripts_dir.is_dir()
    assert calculator.photons_dir.is_dir()
    assert calculator.results_dir.is_dir()


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_application_cleans_working_directory_and_controls_plots(
    monkeypatch, tmp_test_directory, debug, fails, run_context
):
    from simtools.applications import derive_incident_angle as app

    context = run_context.replace(
        args={
            "application_label": "test",
            "telescope": "SSTS-01",
            "debug_plots": debug,
            "model_version": "7.0.0",
        },
        logger=MagicMock(),
    )
    output_dir = context.paths.get_output_directory()
    monkeypatch.setattr(app, "APPLICATION", MagicMock(start=MagicMock(return_value=context)))
    factory = MagicMock()
    factory.return_value.run_for_offsets.return_value = {0: [1]}
    monkeypatch.setattr(app, "IncidentAnglesCalculator", factory)
    plot = MagicMock()
    monkeypatch.setattr(app, "plot_incident_angles", plot)
    cleanup = MagicMock()
    monkeypatch.setattr(app, "cleanup_intermediate_files", cleanup)
    factory.return_value.cleanup_options = {"files": []}
    if fails:
        factory.return_value.run_for_offsets.side_effect = RuntimeError("Simulation failed")
        with pytest.raises(RuntimeError, match="Simulation failed"):
            app.main()
        cleanup.assert_not_called()
        return
    app.main()
    assert plot.call_count == int(debug)
    cleanup.assert_called_once_with(output_dir, files=[])
    factory.return_value.save_model_parameters.assert_called_once()


@pytest.mark.parametrize("mirror_class", [0, 1, 2])
def test_mirror_angles_follow_model(mock_models, config_data, tmp_test_directory, mirror_class):
    mock_models.tel.get_parameter_value.side_effect = None
    mock_models.tel.get_parameter_value.return_value = mirror_class
    calculator = IncidentAnglesCalculator(config_data, tmp_test_directory)
    assert calculator.calculate_primary_secondary_angles is (mirror_class == 2)


def test_run_for_offsets_restores_label_and_collects(monkeypatch, calculator):
    base_label = calculator.label

    def _fake_run(self):
        t = QTable()
        t["angle_incidence_focal"] = [1.0, 2.0] * u.deg
        self.results = t
        return t

    monkeypatch.setattr(ia.IncidentAnglesCalculator, "run", _fake_run)

    offsets = [0.0, 1.5, 3.0]
    res = calculator.run_for_offsets(offsets)
    assert set(res.keys()) == {0.0, 1.5, 3.0}
    assert all(isinstance(v, QTable) for v in res.values())
    assert calculator.label == base_label  # restored


def test_save_model_parameters(calculator, tmp_test_directory, monkeypatch):
    # Mock ModelDataWriter and MetadataCollector
    mock_writer = MagicMock()
    monkeypatch.setattr(ia, "ModelDataWriter", mock_writer)
    monkeypatch.setattr(ia, "MetadataCollector", MagicMock())

    # Create dummy results
    t1 = QTable()
    t1["angle_incidence_focal"] = [1.0, 2.0] * u.deg
    t1["angle_incidence_filter"] = [1.0, 2.0] * u.deg
    t1["angle_incidence_primary"] = [10.0, 20.0] * u.deg
    t1["angle_incidence_secondary"] = [25.0, 35.0] * u.deg

    results_by_offset = {0.0: t1}

    calculator.save_model_parameters(results_by_offset)

    # write_product_data and write_model_parameter are static methods called directly
    assert mock_writer.write_product_data.call_count > 0
    assert mock_writer.write_model_parameter.call_count > 0


@pytest.mark.parametrize("bins", [100, 1000])
def test_calculate_histogram_probability_density(bins):
    """Normalize the integral, including endpoints and ignoring non-finite rays."""
    centers, density = IncidentAnglesCalculator._calculate_histogram(
        [0.0, 0.0, 90.0, float("nan"), float("inf")], bins=bins
    )
    width = 90.0 / bins
    assert len(centers) == bins
    assert centers[[0, -1]] == pytest.approx([width / 2, 90 - width / 2])
    assert sum(density) * width == pytest.approx(1)
    assert density[0] == pytest.approx(2 / (3 * width))
    assert density[-1] == pytest.approx(1 / (3 * width))


def test_build_incidence_distribution_table():
    """Cover _build_incidence_distribution_table static method."""
    calc = object.__new__(IncidentAnglesCalculator)
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    table = calc._build_incidence_distribution_table(data)

    assert isinstance(table, QTable)
    assert "incidence_angle" in table.colnames
    assert "fraction" in table.colnames
    assert table["incidence_angle"].unit == u.deg
    assert len(table) == 100


def test_filter_distribution_includes_rays_without_pixel_hits(calculator, monkeypatch):
    """Rays that miss the pixels (focal angles 30 and 60 deg) count in the distribution."""
    calculator.telescope_model.get_parameter_value.side_effect = None
    calculator.telescope_model.get_parameter_value.return_value = 1
    writer = MagicMock()
    monkeypatch.setattr(ia, "ModelDataWriter", writer)
    monkeypatch.setattr(ia, "MetadataCollector", MagicMock())
    table = QTable({"angle_incidence_focal": [30, 60] * u.deg})
    calculator.save_model_parameters({0: table})
    exported = writer.write_product_data.call_args.kwargs["product_data"]
    nonzero = exported[exported["fraction"] > 0]
    assert list(nonzero["fraction"]) == pytest.approx([0.5 / 0.9, 0.5 / 0.9])
    assert nonzero["incidence_angle"].to_value(u.deg) == pytest.approx([30.15, 59.85])
    assert exported.meta["angle_reference"] == "optical axis at focal surface"


@pytest.fixture
def fake_backend(monkeypatch):
    """Backend returning one photon with angles and mirror hit points."""
    backend = MagicMock()
    backend.return_value.incidence_angles.return_value = (
        {
            "angle_incidence_focal": [36.87],
            "angle_incidence_primary": [10.0],
            "angle_incidence_secondary": [20.0],
            "primary_hit_x": [3.0],
            "primary_hit_y": [4.0],
            "secondary_hit_x": [0.3],
            "secondary_hit_y": [-0.4],
        },
        [Path("run.sh")],
    )
    monkeypatch.setattr(ia.interface, "get_backend", lambda context=None: backend)
    monkeypatch.setattr(ia.simtel_seeds, "instance_seed_pairs", lambda context=None: [(11, 22)])
    return backend


@pytest.mark.parametrize("perfect_mirror", [False, True])
def test_run_with_backend(calculator, fake_backend, perfect_mirror):
    calculator.perfect_mirror = perfect_mirror
    calculator.config_data["off_axis_angle"] = 1.5 * u.deg

    result = calculator.run()

    fake_backend.assert_called_once_with(
        calculator.telescope_model,
        calculator.site_model,
        calculator.label,
        0.0,
        False,
        context=calculator.context,
    )
    request, work_directory, name, with_mirrors = (
        fake_backend.return_value.incidence_angles.call_args.args
    )
    assert (request.off_axis_x, request.off_axis_y, request.source_distance_km) == (1.5, 0, 10)
    assert request.seeds == ((11, 22),)
    assert request.photons_per_instance == 1000
    assert request.parameter_changes == (parameter_changes.PERFECT_MIRROR if perfect_mirror else {})
    assert work_directory == calculator.output_dir
    assert name == f"{calculator.label}_LSTN-01_off1.5"
    assert with_mirrors is calculator.calculate_primary_secondary_angles
    assert Path("run.sh") in calculator._intermediate_files
    assert result.colnames[:3] == [
        "angle_incidence_focal",
        "angle_incidence_primary",
        "angle_incidence_secondary",
    ]
    assert result["primary_hit_radius"][0].to_value(u.m) == pytest.approx(5.0)
    assert result["secondary_hit_y"][0].to_value(u.m) == pytest.approx(-0.4)
    assert (calculator.results_dir / f"incident_angles_{name}.ecsv").exists()


@pytest.mark.parametrize("keep_photons", [False, True])
def test_cleanup_intermediate_files(mock_models, config_data, tmp_test_directory, keep_photons):
    root = Path(tmp_test_directory)
    config_data.update(debug_plots=False, keep_photon_files=keep_photons)
    calc = IncidentAnglesCalculator(config_data, root / "output")
    photons = calc.photons_dir / "photons.lis"
    stars = calc.scripts_dir / "stars.lis"
    for file in (photons, stars):
        file.write_text("data", encoding="utf-8")
    calc._intermediate_files.update([stars] if keep_photons else [stars, photons])
    calc.model_dir.mkdir(parents=True)
    model_file = calc.model_dir / "model.cfg"
    model_file.write_text("model", encoding="utf-8")
    unrelated = calc.scripts_dir / "unrelated.sh"
    unrelated.write_text("unrelated", encoding="utf-8")
    cleanup_intermediate_files(calc.output_dir, **calc.cleanup_options)
    assert photons.exists() is keep_photons
    assert not stars.exists()
    assert not model_file.exists()
    assert unrelated.exists()
    assert calc.results_dir.is_dir()


@pytest.mark.parametrize(("arguments", "expected"), [([], 0), (["--zenith_angle", "40"], 40)])
def test_zenith_angle_cli_default_and_override(arguments, expected):
    from simtools.applications.derive_incident_angle import APPLICATION

    args = APPLICATION.build_parser().parse_args(arguments)
    assert args.zenith_angle.to_value(u.deg) == pytest.approx(expected)


def test_mirror_angle_cli_option_removed():
    from simtools.applications.derive_incident_angle import APPLICATION

    parser = APPLICATION.build_parser()
    assert not hasattr(parser.parse_args([]), "calculate_primary_secondary_angles")
    with pytest.raises(SystemExit):
        parser.parse_args(["--calculate_primary_secondary_angles"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--no-calculate_primary_secondary_angles"])


def test_save_results_no_data_logs_warning(caplog, calculator, tmp_test_directory):
    calculator.results = QTable()  # empty
    caplog.set_level(logging.WARNING, logger=ia.__name__)
    calculator._save_results()
    assert any("No results to save" in rec.message for rec in caplog.records)
    # No file should be created
    out = list(calculator.results_dir.glob("incident_angles_*.ecsv"))
    assert not out


def test_save_model_parameters_no_results_logs_warning(
    caplog, calculator, tmp_test_directory, monkeypatch
):
    monkeypatch.setattr(
        ia,
        "ModelDataWriter",
        MagicMock(side_effect=AssertionError("ModelDataWriter should not be called")),
    )

    caplog.set_level(logging.WARNING, logger=ia.__name__)
    calculator.save_model_parameters({0.0: QTable()})
    assert any("No results to write model parameters." in rec.message for rec in caplog.records)

    assert not list(Path(tmp_test_directory).glob("*.ecsv"))
    assert not list(Path(tmp_test_directory).glob("*.json"))


def test_source_distance_km_with_plain_float(calculator):
    """Cover _source_distance_km when source_distance is a plain float."""
    calculator.config_data["source_distance"] = 15.0
    assert calculator._source_distance_km() == pytest.approx(15.0)


def test_calculate_histogram_empty_after_filtering():
    """Cover _calculate_histogram when all data is filtered out as non-finite."""
    bin_centers, hist = IncidentAnglesCalculator._calculate_histogram(
        [float("nan"), float("inf"), float("-inf")], bins=10
    )
    assert len(bin_centers) == 0
    assert len(hist) == 0


def test_save_model_parameters_mirror_class_2(tmp_test_directory, monkeypatch, mock_models):
    """Cover save_model_parameters with mirror_class == 2 (dual-mirror telescope)."""
    mock_writer = MagicMock()
    monkeypatch.setattr(ia, "ModelDataWriter", mock_writer)
    monkeypatch.setattr(ia, "MetadataCollector", MagicMock())

    config_data = {
        "telescope": "SSTS-01",
        "site": "South",
        "model_version": "7.0.0",
        "parameter_version": "1.0.0",
    }

    calculator = IncidentAnglesCalculator(
        config_data=config_data,
        output_dir=tmp_test_directory,
        label="test",
    )

    calculator.telescope_model = MagicMock()
    calculator.telescope_model.get_parameter_value.return_value = 2

    t1 = QTable()
    t1["angle_incidence_focal"] = [1.0, 2.0, 3.0] * u.deg
    t1["angle_incidence_filter"] = [1.0, 2.0, 3.0] * u.deg
    t1["angle_incidence_primary"] = [10.0, 20.0, 30.0] * u.deg
    t1["angle_incidence_secondary"] = [5.0, 10.0, 15.0] * u.deg

    results_by_offset = {0.0: t1}
    calculator.save_model_parameters(results_by_offset)

    assert mock_writer.write_product_data.call_count >= 3
    assert mock_writer.write_model_parameter.call_count >= 3


def test_save_model_parameters_no_results(tmp_test_directory, monkeypatch, caplog, mock_models):
    """Cover early return in save_model_parameters when no results."""
    mock_writer = MagicMock()
    monkeypatch.setattr(ia, "ModelDataWriter", mock_writer)
    monkeypatch.setattr(ia, "MetadataCollector", MagicMock())

    config_data = {
        "telescope": "LSTN-01",
        "site": "North",
        "model_version": "7.0.0",
    }

    calculator = IncidentAnglesCalculator(
        config_data=config_data,
        output_dir=tmp_test_directory,
        label="test",
    )
    calculator.telescope_model = MagicMock()
    calculator.telescope_model.get_parameter_value.return_value = 1

    caplog.set_level(logging.WARNING, logger=ia.__name__)
    calculator.save_model_parameters({})

    assert any("No results to write model parameters." in rec.message for rec in caplog.records)
    assert mock_writer.write_product_data.call_count == 0


@pytest.mark.parametrize("mirror_class", [0, 1, 2])
def test_exports_only_incidence_parameters(calculator, monkeypatch, mirror_class):
    """Keep the laboratory lightguide file intact while exporting ray-traced angles."""
    calculator.telescope_model.get_parameter_value.side_effect = None
    calculator.telescope_model.get_parameter_value.return_value = mirror_class
    writer = MagicMock()
    monkeypatch.setattr(ia, "ModelDataWriter", writer)
    monkeypatch.setattr(ia, "MetadataCollector", MagicMock())
    parameter_dir = calculator.output_dir / calculator.config_data["telescope"]
    parameter_dir.mkdir()
    lab_file = parameter_dir / "lightguide_efficiency_vs_incidence_angle.ecsv"
    original_data = "# laboratory measurement\n0 0.85\n30 0.72\n"
    lab_file.write_text(original_data, encoding="utf-8")
    table = QTable()
    for surface in ("focal", "primary", "secondary"):
        table[f"angle_incidence_{surface}"] = [10, 20, 30] * u.deg

    calculator.save_model_parameters({0.0: table})

    expected = ["camera_filter_photon_incident_angle"]
    if mirror_class == 2:
        expected.extend(["primary_mirror_incidence_angle", "secondary_mirror_incidence_angle"])
    assert [
        call.kwargs["parameter_name"] for call in writer.write_model_parameter.call_args_list
    ] == expected
    assert [
        call.kwargs["output_file"].name for call in writer.write_product_data.call_args_list
    ] == [f"{name}.ecsv" for name in expected]
    for call in writer.write_product_data.call_args_list:
        table = call.kwargs["product_data"]
        assert table.colnames == ["incidence_angle", "fraction"]
        assert table["incidence_angle"].unit == u.deg
        assert table.meta["normalization"] == "probability density per degree"
        assert sum(table["fraction"]) * table.meta["bin_width_deg"] == pytest.approx(1)
    assert lab_file.read_text(encoding="utf-8") == original_data

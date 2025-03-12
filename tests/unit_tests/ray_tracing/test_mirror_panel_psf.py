#!/usr/bin/python3

import copy
import re
from unittest.mock import patch

import astropy.units as u
import pytest

from simtools.ray_tracing.mirror_panel_psf import MirrorPanelPSF


@pytest.fixture
def mock_args_dict(tmp_test_directory):
    return {
        "test": False,
        "psf_measurement": "tests/resources/MLTdata-preproduction.ecsv",
        "psf_measurement_containment_mean": 0.5,
        "psf_measurement_containment_sigma": 0.1,
        "site": "North",
        "telescope": "LSTN-01",
        "model_version": "test_version",
        "mirror_list": None,
        "random_focal_length": None,
        "no_tuning": False,
        "rnda": 0,
        "rtol_psf_containment": 0.1,
        "simtel_path": "path/to/simtel",
        "number_of_mirrors_to_test": 2,
        "use_random_focal_length": False,
        "containment_fraction": 0.8,
        "output_path": tmp_test_directory,
    }


@pytest.fixture
def mock_telescope_model_string():
    return "simtools.ray_tracing.mirror_panel_psf.TelescopeModel"


@pytest.fixture
def mock_find_file_string():
    return "simtools.ray_tracing.mirror_panel_psf.gen.find_file"


@pytest.fixture
def mock_run_simulations_and_analysis_string():
    return "simtools.ray_tracing.mirror_panel_psf.MirrorPanelPSF.run_simulations_and_analysis"


@pytest.fixture
def mock_mirror_panel_psf(mock_args_dict, mock_telescope_model_string, mock_find_file_string):
    with (
        patch(mock_telescope_model_string),
        patch(mock_find_file_string),
    ):
        db_config = {"db": "config"}
        label = "test_label"
        mirror_panel_psf = MirrorPanelPSF(label, mock_args_dict, db_config)
        yield mirror_panel_psf


def test_define_telescope_model(mock_args_dict, mock_telescope_model_string, mock_find_file_string):
    args_dict = copy.deepcopy(mock_args_dict)
    # no mirror list, no random focal length
    with (
        patch(mock_telescope_model_string) as mock_telescope_model,
        patch(mock_find_file_string) as mock_find_file,
    ):
        args_dict["mirror_list"] = None
        args_dict["random_focal_length"] = None
        db_config = {"db": "config"}
        label = "test_label"

        mirror_panel_psf = MirrorPanelPSF(label, args_dict, db_config)
        tel = mirror_panel_psf.telescope_model

        mock_telescope_model.assert_called_once_with(
            site=args_dict["site"],
            telescope_name=args_dict["telescope"],
            model_version=args_dict["model_version"],
            mongo_db_config=db_config,
            label=label,
        )
        tel.export_model_files.assert_called_once()
        mock_find_file.assert_not_called()
        tel.change_parameter.assert_not_called()
        tel.export_parameter_file.assert_not_called()

    # mirror list and random focal length
    with (
        patch(mock_telescope_model_string) as mock_telescope_model,
        patch(mock_find_file_string) as mock_find_file,
    ):
        args_dict["mirror_list"] = "mirror_list_CTA-N-LST1_v2019-03-31_rotated.ecsv"
        args_dict["model_path"] = "tests/resources"
        args_dict["random_focal_length"] = 0.1
        db_config = {"db": "config"}
        label = "test_label"

        mirror_panel_psf = MirrorPanelPSF(label, args_dict, db_config)
        tel = mirror_panel_psf.telescope_model

        mock_telescope_model.assert_called_once_with(
            site=args_dict["site"],
            telescope_name=args_dict["telescope"],
            model_version=args_dict["model_version"],
            mongo_db_config=db_config,
            label=label,
        )
        tel.export_model_files.assert_called_once()
        mock_find_file.assert_called_once()
        assert tel.change_parameter.call_count == 2
        tel.export_parameter_file.assert_called_once()


def test_define_telescope_model_test_errors(
    mock_args_dict, mock_telescope_model_string, mock_find_file_string
):
    args_dict = copy.deepcopy(mock_args_dict)
    # test mode, missing PSF measurement
    with (
        patch(mock_telescope_model_string),
        patch(mock_find_file_string),
    ):
        db_config = {"db": "config"}
        label = "test_label"

        args_dict["mirror_list"] = "mirror_list_CTA-N-LST1_v2019-03-31_rotated.ecsv"
        args_dict["model_path"] = "tests/resources"
        args_dict["random_focal_length"] = 0.1
        args_dict["test"] = True

        mirror_panel_psf = MirrorPanelPSF(label, args_dict, db_config)

        assert mirror_panel_psf.args_dict["number_of_mirrors_to_test"] == 2

        args_dict["psf_measurement"] = None
        args_dict["psf_measurement_containment_mean"] = None
        with pytest.raises(ValueError, match=r"Missing PSF measurement"):
            MirrorPanelPSF(label, args_dict, db_config)


def test_write_optimization_data(mock_mirror_panel_psf):
    mirror_panel_psf = copy.deepcopy(mock_mirror_panel_psf)
    mirror_panel_psf.results_rnda = [0.1, 0.2, 0.3]
    mirror_panel_psf.results_mean = [0.4, 0.5, 0.6]
    mirror_panel_psf.results_sig = [0.01, 0.02, 0.03]
    mirror_panel_psf.rnda_opt = 0.25
    mirror_panel_psf.mean_d80 = 0.55
    mirror_panel_psf.sig_d80 = 0.025

    with (
        patch("simtools.ray_tracing.mirror_panel_psf.writer.ModelDataWriter.dump") as mock_dump,
        patch("simtools.ray_tracing.mirror_panel_psf.MetadataCollector") as mock_metadata_collector,
    ):
        mirror_panel_psf.write_optimization_data()
        mock_dump.assert_called_once()
        mock_metadata_collector.assert_called_once()


def test_run_simulations_and_analysis(mock_telescope_model_string, mock_find_file_string):
    # Not using pytest.fixtures to run test in isolation (not using the same instance of the class)
    rnda = 0.1
    args_dict = {
        "test": False,
        "psf_measurement": "tests/resources/MLTdata-preproduction.ecsv",
        "psf_measurement_containment_mean": 0.5,
        "psf_measurement_containment_sigma": 0.1,
        "site": "North",
        "telescope": "LSTN-01",
        "model_version": "test_version",
        "mirror_list": None,
        "random_focal_length": None,
        "no_tuning": False,
        "rnda": 0,
        "simtel_path": "path/to/simtel",
        "number_of_mirrors_to_test": 2,
        "use_random_focal_length": False,
        "containment_fraction": 0.8,
        "output_path": "",
    }

    with (
        patch(mock_telescope_model_string),
        patch(mock_find_file_string),
        patch("simtools.ray_tracing.mirror_panel_psf.RayTracing") as mock_ray_tracing,
    ):
        db_config = {"db": "config"}
        label = "test_label"
        mirror_panel_psf = MirrorPanelPSF(label, args_dict, db_config)

        mock_ray_instance = mock_ray_tracing.return_value
        mock_ray_instance.get_mean.return_value = 0.5 * u.cm
        mock_ray_instance.get_std_dev.return_value = 0.1 * u.cm

        mean_d80, sig_d80 = mirror_panel_psf.run_simulations_and_analysis(rnda)

        mirror_panel_psf.telescope_model.change_parameter.assert_called_once_with(
            "mirror_reflection_random_angle", rnda
        )
        mock_ray_tracing.assert_called_once_with(
            telescope_model=mirror_panel_psf.telescope_model,
            simtel_path=mirror_panel_psf.args_dict.get("simtel_path", None),
            single_mirror_mode=True,
            mirror_numbers=(
                list(range(1, mirror_panel_psf.args_dict["number_of_mirrors_to_test"] + 1))
                if mirror_panel_psf.args_dict["test"]
                else "all"
            ),
            use_random_focal_length=mirror_panel_psf.args_dict["use_random_focal_length"],
        )
        mock_ray_instance.simulate.assert_called_once_with(
            test=mirror_panel_psf.args_dict["test"], force=True
        )
        mock_ray_instance.analyze.assert_called_once_with(force=True)
        assert mean_d80 == 0.5
        assert sig_d80 == 0.1


def test_print_results(mock_mirror_panel_psf, capsys):
    mirror_panel_psf = copy.deepcopy(mock_mirror_panel_psf)
    mirror_panel_psf.results_rnda = [0.1, 0.2, 0.3]
    mirror_panel_psf.results_mean = [0.4, 0.5, 0.6]
    mirror_panel_psf.results_sig = [0.01, 0.02, 0.03]
    mirror_panel_psf.rnda_opt = 0.25
    mirror_panel_psf.rnda_start = 0.3
    mirror_panel_psf.mean_d80 = 0.55
    mirror_panel_psf.sig_d80 = 0.025

    mirror_panel_psf.print_results()
    out = capsys.readouterr().out
    assert "StdDev" in out
    assert "New value" in out

    mirror_panel_psf.args_dict["psf_measurement_containment_sigma"] = None
    mirror_panel_psf.print_results()
    out = capsys.readouterr().out
    assert "StdDev = 0.010 cm" not in out
    assert "New value" in out


def test_get_starting_value_from_args(mock_mirror_panel_psf, caplog):
    mirror_psf = copy.deepcopy(mock_mirror_panel_psf)
    mirror_psf.args_dict["rnda"] = 0.5
    with caplog.at_level("INFO"):
        rnda_start = mirror_psf._get_starting_value()
    assert rnda_start == 0.5
    assert "Start value for mirror_reflection_random_angle: 0.5 deg" in caplog.text


def test_get_starting_value_from_model(mock_mirror_panel_psf, caplog):
    mirror_psf = copy.deepcopy(mock_mirror_panel_psf)
    mirror_psf.args_dict["rnda"] = 0
    mirror_psf.telescope_model.get_parameter_value = patch(
        "simtools.ray_tracing.mirror_panel_psf.TelescopeModel.get_parameter_value",
        return_value=[0.3],
    ).start()
    with caplog.at_level("INFO"):
        rnda_start = mirror_psf._get_starting_value()
    assert rnda_start == 0.3
    assert "Start value for mirror_reflection_random_angle: 0.3 deg" in caplog.text


def test_derive_random_reflection_angle_no_tuning(
    mock_mirror_panel_psf, mock_run_simulations_and_analysis_string
):
    mirror_psf = copy.deepcopy(mock_mirror_panel_psf)
    mirror_psf.args_dict["no_tuning"] = True
    mirror_psf.rnda_start = 0.1
    with patch(
        mock_run_simulations_and_analysis_string,
        return_value=(0.5, 0.1),
    ) as mock_run_simulations_and_analysis:
        mock_run_simulations_and_analysis.start()
        mirror_psf.derive_random_reflection_angle()

        mock_run_simulations_and_analysis.assert_called_once_with(0.1, save_figures=False)
        assert mirror_psf.mean_d80 == 0.5
        assert mirror_psf.sig_d80 == 0.1


def test_derive_random_reflection_angle_with_tuning(
    mock_mirror_panel_psf, mock_run_simulations_and_analysis_string
):
    mirror_psf = copy.deepcopy(mock_mirror_panel_psf)
    mirror_psf.args_dict["no_tuning"] = False
    mirror_psf.rnda_start = 0.1

    with (
        patch(
            "simtools.ray_tracing.mirror_panel_psf.MirrorPanelPSF._optimize_reflection_angle"
        ) as mock_optimize,
        patch(
            mock_run_simulations_and_analysis_string,
            return_value=(0.5, 0.1),
        ) as mock_run_simulations,
    ):
        mirror_psf.derive_random_reflection_angle()

        mock_optimize.assert_called_once()
        mock_run_simulations.assert_called_once_with(mirror_psf.rnda_opt, save_figures=False)
        assert mirror_psf.mean_d80 == 0.5
        assert mirror_psf.sig_d80 == 0.1


def test_optimize_reflection_angle(mock_mirror_panel_psf, mock_run_simulations_and_analysis_string):
    mirror_psf = copy.deepcopy(mock_mirror_panel_psf)
    mirror_psf.rnda_start = 0.1
    mirror_psf.args_dict["psf_measurement_containment_mean"] = 0.5

    with patch(
        mock_run_simulations_and_analysis_string,
        side_effect=[
            (0.7, 0.1),  # First call
            (0.6, 0.1),  # Second call
            (0.45, 0.1),  # Third call
        ],
    ) as mock_run_simulations:
        mirror_psf._optimize_reflection_angle()

        assert mirror_psf.results_rnda == pytest.approx([0.1, 0.09, 0.08])
        assert mirror_psf.results_mean == pytest.approx([0.7, 0.6, 0.45])
        assert mirror_psf.results_sig == pytest.approx([0.1, 0.1, 0.1])
        assert mock_run_simulations.call_count == 3

    with patch(
        mock_run_simulations_and_analysis_string,
        side_effect=lambda *args, **kwargs: (0.6, 0.1),
    ) as mock_run_simulations:
        with pytest.raises(
            ValueError,
            match=re.escape("Maximum iterations (100) reached without convergence."),
        ):
            mirror_psf._optimize_reflection_angle()

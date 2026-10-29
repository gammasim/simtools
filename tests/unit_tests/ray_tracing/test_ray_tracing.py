#!/usr/bin/python3

import copy
import logging
from math import pi, tan
from pathlib import Path
from unittest.mock import call

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable

from simtools.constants import SIMTEL_MAX_SEED
from simtools.ray_tracing.ray_tracing import INVALID_KEY_TO_PLOT, RayTracing
from simtools.utils import random

BACKEND_METADATA = "ray_tracing_backend"
PROCESS_IMAGES = "_process_off_axis_and_mirror"
CACHED_COLUMN = "cached"


def _example_rows():
    return [
        (
            0.0 * u.deg,
            0.0 * u.deg,
            0.0 * u.deg,
            4.256768651160611 * u.cm,
            0.1 * u.deg,
            100.0 * u.m * u.m,
            0.01 * u.cm,
            0.02 * u.cm,
            200.0,
            2.0,
        ),
        (
            0.0 * u.deg,
            2.0 * u.deg,
            2.0 * u.deg,
            4.356768651160611 * u.cm,
            0.2 * u.deg,
            110.0 * u.m * u.m,
            0.03 * u.cm,
            0.04 * u.cm,
            210.0,
            3.0,
        ),
    ]


def _set_context_args(ray, args):
    """Set read-only arguments on the test run context."""
    ray.context = ray.context.replace(args=args)


def _example_rows_single_mirror():
    return [
        (*row, mirror_number) for row, mirror_number in zip(_example_rows(), [1, 2], strict=True)
    ]


@pytest.fixture
def test_photons_file():
    return Path(
        "ray_tracing_photons_North_LSTN-01_d10.0km_za20.0deg_off0.000deg_validate_optics.lis.gz"
    )


@pytest.fixture
def off_axis_string():
    return "Off-axis angle"


@pytest.fixture
def telescope_model_lst_mock(mocker, tmp_test_directory, io_handler):
    mock_telescope_model = mocker.Mock()
    mock_telescope_model.mirrors.number_of_mirrors = 3
    mock_telescope_model.get_parameter_value.return_value = 10.0
    mock_telescope_model.site = "North"
    mock_telescope_model.name = "LSTN-01"
    mock_telescope_model.label = "ray_tracing"
    mock_telescope_model.write_sim_telarray_config_file = mocker.Mock()
    mock_telescope_model.mirrors.get_single_mirror_parameters = mocker.Mock(
        return_value=(0, 0, 2600.0, 0, 0)
    )

    mock_telescope_model.config_file_path = tmp_test_directory.join("model")
    return mock_telescope_model


@pytest.fixture
def ray_tracing_lst(telescope_model_lst_mock, site_model_north, run_context):
    """A RayTracing instance with a local output directory."""

    ray_tracing_lst = RayTracing(
        telescope_model=telescope_model_lst_mock,
        site_model=site_model_north,
        label="validate_optics",
        source_distance=10 * u.km,
        zenith_angle=20 * u.deg,
        off_axis_angle=[0] * u.deg,
        offset_directions=["N"],
        context=run_context,
    )

    ray_tracing_lst.output_directory.mkdir(parents=True, exist_ok=True)
    return ray_tracing_lst


@pytest.fixture
def ray_tracing_lst_single_mirror_mode(telescope_model_lst_mock, site_model_north, run_context):
    telescope_model_lst_mock.write_sim_telarray_config_file()
    return RayTracing(
        telescope_model=telescope_model_lst_mock,
        site_model=site_model_north,
        label="validate_optics",
        source_distance=10 * u.km,
        zenith_angle=20 * u.deg,
        off_axis_angle=[0] * u.deg,
        single_mirror_mode=True,
        mirror_numbers=[0, 2],
        offset_directions=["N"],
        context=run_context,
    )


def test_ray_tracing_init(telescope_model_lst_mock, site_model_north, caplog, run_context):
    with caplog.at_level(logging.DEBUG):
        ray = RayTracing(
            telescope_model=telescope_model_lst_mock,
            site_model=site_model_north,
            label=telescope_model_lst_mock.label,
            zenith_angle=30 * u.deg,
            source_distance=10 * u.km,
            off_axis_angle=[0, 2] * u.deg,
            offset_directions=["N"],
            context=run_context,
        )

    assert ray.zenith_angle == pytest.approx(30)
    assert len(ray.off_axis_angle) == 2


def test_ray_tracing_single_mirror_mode(
    telescope_model_lst_mock, site_model_north, caplog, run_context
):
    telescope_model_lst_mock.write_sim_telarray_config_file()

    with caplog.at_level(logging.DEBUG):
        ray = RayTracing(
            telescope_model=telescope_model_lst_mock,
            site_model=site_model_north,
            label=telescope_model_lst_mock.label,
            zenith_angle=30 * u.deg,
            source_distance=10 * u.km,
            off_axis_angle=[0, 2] * u.deg,
            single_mirror_mode=True,
            mirror_numbers="all",
            offset_directions=["N"],
            context=run_context,
        )

    assert ray.zenith_angle == pytest.approx(30)
    assert len(ray.off_axis_angle) == 2
    assert ray.single_mirror_mode
    assert "Single mirror mode is activated" in caplog.text
    assert len(ray.mirrors) == telescope_model_lst_mock.mirrors.number_of_mirrors


def test_ray_tracing_single_mirror_mode_mirror_numbers(
    telescope_model_lst_mock, site_model_north, mocker, run_context
):
    telescope_model_lst_mock.write_sim_telarray_config_file()
    ray = RayTracing(
        telescope_model=telescope_model_lst_mock,
        site_model=site_model_north,
        label=telescope_model_lst_mock.label,
        source_distance=10 * u.km,
        zenith_angle=30 * u.deg,
        off_axis_angle=[0, 2] * u.deg,
        single_mirror_mode=True,
        mirror_numbers=[1, 2, 3],
        context=run_context,
    )

    assert list(ray.mirrors.keys()) == [1, 2, 3]

    mocker.patch.object(
        ray.telescope_model.mirrors, "get_single_mirror_parameters", return_value=(0, 0, 0, 0, 0)
    )
    mocker.patch.object(ray, "_get_mirror_panel_focal_length", return_value=0)
    with pytest.raises(ValueError, match=r"Focal length is invalid \(NaN or close to zero\)"):
        ray._initialize_single_mirror_mode([1, 2, 3])


def test_export_results(ray_tracing_lst, caplog, mocker):

    ray = copy.deepcopy(ray_tracing_lst)
    ray.export_results()
    assert "No results to export" in caplog.text

    _rows = _example_rows()
    ray._store_results(_rows)
    mock_write = mocker.patch("astropy.io.ascii.write")
    with caplog.at_level(logging.INFO):
        ray.export_results()
    assert "Exporting results" in caplog.text
    mock_write.assert_called_once_with(
        ray._results, ray._file_results, format="ecsv", overwrite=True
    )


def test_ray_tracing_no_images(ray_tracing_lst, caplog):

    with caplog.at_level("WARNING"):
        assert ray_tracing_lst.images() is None
    assert "No image found" in caplog.text


@pytest.mark.parametrize(
    ("export", "force", "read_called", "store_called", "export_called"),
    [
        (True, False, True, False, True),
        (True, True, False, True, True),
        (False, False, True, False, False),
    ],
)
def test_analyze(ray_tracing_lst, mocker, export, force, read_called, store_called, export_called):
    ray_tracing_lst._results = QTable(meta={BACKEND_METADATA: "sim_telarray"})
    mock_read_results = mocker.patch.object(ray_tracing_lst, "_read_results")
    mock_store_results = mocker.patch.object(ray_tracing_lst, "_store_results")
    mock_export_results = mocker.patch.object(ray_tracing_lst, "export_results")
    mock_exists = mocker.patch("pathlib.Path.exists", return_value=True)
    mock_process = mocker.patch.object(ray_tracing_lst, PROCESS_IMAGES)

    ray_tracing_lst.analyze(export=export, force=force)

    mock_exists.assert_called_once()
    mock_process.assert_called_once()

    assert mock_read_results.called == read_called
    assert mock_store_results.called == store_called
    assert mock_export_results.called == export_called


@pytest.mark.parametrize("cached_backend", [None, "obdeect"])
def test_analyze_rejects_results_from_another_backend(ray_tracing_lst, mocker, cached_backend):
    cached = QTable({CACHED_COLUMN: [True]})
    if cached_backend is not None:
        cached.meta[BACKEND_METADATA] = cached_backend
    cached.write(ray_tracing_lst._file_results, format="ascii.ecsv")
    mocker.patch.object(ray_tracing_lst, PROCESS_IMAGES, return_value=_example_rows())

    ray_tracing_lst.analyze(export=False)

    assert CACHED_COLUMN not in ray_tracing_lst._results.colnames
    assert BACKEND_METADATA in ray_tracing_lst._results.meta


def test_images_with_psf_images(ray_tracing_lst, mocker):
    mock_psf_image = mocker.Mock()
    # off_axis_angle with offset_directions=["N"] results in [(0.0, 0.0)] since it's (0, 1*0) = (0, 0)
    ray_tracing_lst.psf_images = {(0.0, 0.0): mock_psf_image}

    images = ray_tracing_lst.images()

    assert images is not None
    assert len(images) == 1
    assert images[0] == mock_psf_image


def test_store_results(ray_tracing_lst, ray_tracing_lst_single_mirror_mode, off_axis_string):
    _rows = _example_rows()
    ray_tracing_lst._store_results(_rows)

    assert isinstance(ray_tracing_lst._results, QTable)
    assert len(ray_tracing_lst._results) == len(_rows)
    assert ray_tracing_lst._results.colnames == [
        "off_x",
        "off_y",
        "off axis angle",
        "psf_cm",
        "psf_deg",
        "eff_area",
        "centroid_x_err",
        "centroid_y_err",
        "eff_flen",
        "eff_flen_err",
    ]

    # single mirror mode
    _rows = _example_rows_single_mirror()
    ray_tracing_lst_single_mirror_mode._store_results(_rows)
    assert len(ray_tracing_lst_single_mirror_mode._results) == len(_rows)


def test_get_mirror_panel_focal_length_with_random_normal(ray_tracing_lst, mocker):
    mock_get_parameter_value = mocker.patch.object(
        ray_tracing_lst.telescope_model, "get_parameter_value", side_effect=[10.0, [1.0, 0.0]]
    )
    mock_rng = mocker.patch("numpy.random.default_rng")
    mock_rng_instance = mock_rng.return_value
    mock_rng_instance.normal.return_value = 2.0
    ray_tracing_lst.use_random_focal_length = True

    focal_length = ray_tracing_lst._get_mirror_panel_focal_length()

    assert focal_length == pytest.approx(12.0)
    mock_get_parameter_value.assert_has_calls(
        [call("mirror_focal_length"), call("random_focal_length")]
    )
    mock_rng_instance.normal.assert_called_once_with(loc=0, scale=1.0)


def test_get_mirror_panel_focal_length_with_random_uniform(ray_tracing_lst, mocker):
    mock_get_parameter_value = mocker.patch.object(
        ray_tracing_lst.telescope_model, "get_parameter_value", side_effect=[10.0, [0.0, 1.0]]
    )
    mock_rng = mocker.patch("numpy.random.default_rng")
    mock_rng_instance = mock_rng.return_value
    mock_rng_instance.uniform.return_value = 0.5
    ray_tracing_lst.use_random_focal_length = True

    focal_length = ray_tracing_lst._get_mirror_panel_focal_length()

    assert focal_length == pytest.approx(10.5)
    mock_get_parameter_value.assert_has_calls(
        [call("mirror_focal_length"), call("random_focal_length")]
    )
    mock_rng_instance.uniform.assert_called_once_with(low=-1.0, high=1.0)


@pytest.mark.parametrize("max_workers", [0, -1])
def test_ray_tracing_simulate_rejects_non_positive_worker_count(ray_tracing_lst, max_workers):
    with pytest.raises(ValueError, match="max_workers must be a positive integer"):
        ray_tracing_lst.simulate(max_workers=max_workers)


def test_analyze_image(ray_tracing_lst, mocker):
    mock_image = mocker.Mock()
    mock_image.get_psf.side_effect = [5.0, 0.1]
    mock_image.get_effective_area.return_value = 100.0
    mock_image.centroid_x = 0.5
    mock_image.centroid_y = 0.2
    mock_image.centroid_x_error = 0.05
    mock_image.centroid_y_error = 0.02

    off_x = 1.0
    off_y = 1.5
    theta_offset = 2.0
    containment_fraction = 0.8
    tel_transmission = 0.9

    result = ray_tracing_lst._analyze_image(
        image=mock_image,
        off_x=off_x,
        off_y=off_y,
        theta_offset=theta_offset,
        containment_fraction=containment_fraction,
        tel_transmission=tel_transmission,
    )

    radius = np.hypot(mock_image.centroid_x, mock_image.centroid_y)
    radius_error = np.sqrt(
        ((mock_image.centroid_x / radius) * mock_image.centroid_x_error) ** 2
        + ((mock_image.centroid_y / radius) * mock_image.centroid_y_error) ** 2
    )
    expected_eff_flen = radius / tan(theta_offset * pi / 180.0)
    expected_eff_flen_error = radius_error / tan(theta_offset * pi / 180.0)

    assert result == (
        1.0 * u.deg,
        1.5 * u.deg,
        2.0 * u.deg,
        5.0 * u.cm,
        0.1 * u.deg,
        100.0 * u.m * u.m,
        0.05 * u.cm,
        0.02 * u.cm,
        expected_eff_flen,
        expected_eff_flen_error,
    )
    mock_image.get_psf.assert_has_calls(
        [call(containment_fraction, "cm"), call(containment_fraction, "deg")]
    )
    mock_image.get_effective_area.assert_called_once_with(tel_transmission)


def test_analyze_image_zero_theta_offset(ray_tracing_lst, mocker):
    mock_image = mocker.Mock()
    mock_image.get_psf.side_effect = [5.0, 0.1]
    mock_image.get_effective_area.return_value = 100.0
    mock_image.centroid_x = 0.5
    mock_image.centroid_y = 0.2
    mock_image.centroid_x_error = 0.05
    mock_image.centroid_y_error = 0.02

    result = ray_tracing_lst._analyze_image(
        image=mock_image,
        off_x=0.0,
        off_y=0.0,
        theta_offset=0.0,
        containment_fraction=0.8,
        tel_transmission=0.9,
    )

    assert np.isnan(result[-2])
    assert np.isnan(result[-1])


def test_read_results(ray_tracing_lst, mocker):
    mock_read = mocker.patch("astropy.io.ascii.read", return_value=QTable())
    ray_tracing_lst._file_results = Path("dummy_path.ecsv")

    ray_tracing_lst._read_results()

    mock_read.assert_called_once_with(ray_tracing_lst._file_results, format="ecsv")
    assert isinstance(ray_tracing_lst._results, QTable)


def test_get_psf_mm_raises_when_no_results(ray_tracing_lst):
    ray = copy.deepcopy(ray_tracing_lst)
    ray._results = None
    with pytest.raises(RuntimeError, match=r"run analyze\(\) first"):
        ray.get_psf_mm()


def test_get_psf_mm_returns_mm_for_quantity(ray_tracing_lst):
    ray = copy.deepcopy(ray_tracing_lst)
    ray._results = QTable(
        {
            "psf_cm": [1.5 * u.cm],
        }
    )
    assert pytest.approx(ray.get_psf_mm()) == 15


def test_get_psf_mm_returns_mm_for_plain_float(ray_tracing_lst):
    ray = copy.deepcopy(ray_tracing_lst)
    ray._results = QTable(
        {
            "psf_cm": [1.5],
        }
    )
    assert pytest.approx(ray.get_psf_mm()) == 15


def test_plot_histogram_valid_key(ray_tracing_lst, mocker):
    mock_plot_histogram = mocker.patch("simtools.ray_tracing.ray_tracing.visualize.plot_histogram")

    ray_tracing_lst._results = QTable(
        {
            "psf_cm": [4.256768651160611, 4.356768651160611],
            "psf_deg": [0.1, 0.2],
            "eff_area": [100.0, 110.0],
            "eff_flen": [200.0, 210.0],
        }
    )

    ray_tracing_lst.plot_histogram(key="psf_cm", bins=10)

    mock_plot_histogram.assert_called_once_with(ray_tracing_lst._results["psf_cm"], bins=10)


def test_plot_histogram_invalid_key(ray_tracing_lst):
    ray_tracing_lst._results = QTable(
        {
            "psf_cm": [4.256768651160611, 4.356768651160611],
            "psf_deg": [0.1, 0.2],
            "eff_area": [100.0, 110.0],
            "eff_flen": [200.0, 210.0],
        }
    )

    with pytest.raises(KeyError, match=INVALID_KEY_TO_PLOT):
        ray_tracing_lst.plot_histogram(key="invalid_key", bins=10)


def test_plot_valid_key(ray_tracing_lst, mocker):
    mock_generate_file_name = mocker.patch.object(
        ray_tracing_lst, "_generate_file_name", return_value="plot_file.pdf"
    )

    ray_tracing_lst._results = QTable(
        {
            "off axis angle": [0.0, 2.0],
            "psf_cm": [4.256768651160611, 4.356768651160611],
            "psf_deg": [0.1, 0.2],
            "eff_area": [100.0, 110.0],
            "eff_flen": [200.0, 210.0],
            "eff_flen_err": [2.0, 3.0],
        }
    )

    ray_tracing_lst.plot(key="psf_cm", save=True)

    mock_generate_file_name.assert_called_once_with(
        file_type="ray_tracing",
        suffix=".pdf",
        extra_label="psf_cm",
    )
    assert ray_tracing_lst.output_directory.joinpath("figures", "plot_file.pdf").exists()


def test_plot_effective_focal_length_includes_error_column(ray_tracing_lst, mocker):
    mock_plot_1d = mocker.patch("simtools.ray_tracing.ray_tracing.visualize.plot_1d")
    ray_tracing_lst._results = QTable(
        {
            "off axis angle": [0.0, 2.0],
            "eff_flen": [200.0, 210.0],
            "eff_flen_err": [2.0, 3.0],
        }
    )

    ray_tracing_lst.plot(
        key="eff_flen",
        error_type="errorbar",
        error_label="Uncertainty",
        no_legend=False,
    )

    data = mock_plot_1d.call_args.args[0]
    table = data["Effective focal length"]
    assert table.colnames == ["off axis angle", "eff_flen", "eff_flen_err"]
    assert mock_plot_1d.call_args.kwargs["error_type"] == "errorbar"
    assert mock_plot_1d.call_args.kwargs["error_label"] == "Uncertainty"
    assert mock_plot_1d.call_args.kwargs["no_legend"] is False


def test_plot_invalid_key(ray_tracing_lst, off_axis_string):
    ray_tracing_lst._results = QTable(
        {
            off_axis_string: [0.0, 2.0],
            "psf_cm": [4.256768651160611, 4.356768651160611],
            "psf_deg": [0.1, 0.2],
            "eff_area": [100.0, 110.0],
            "eff_flen": [200.0, 210.0],
        }
    )

    with pytest.raises(KeyError, match=INVALID_KEY_TO_PLOT):
        ray_tracing_lst.plot(key="invalid_key", save=True)


def test_plot_save_writes_psf_images_and_cumulative(ray_tracing_lst, mocker):

    mock_visualize_plot_table = mocker.patch(
        "simtools.ray_tracing.ray_tracing.visualize.plot_table"
    )
    mock_plot = mock_visualize_plot_table.return_value
    mocker.patch.object(mock_plot, "savefig")

    # Avoid real filesystem writes; only validate paths passed to methods.
    mocker.patch("pathlib.Path.mkdir")

    def _fake_name(*, file_type, suffix, off_axis_x=None, off_axis_y=None, extra_label=None):
        assert file_type == "ray_tracing"
        assert suffix == ".pdf"
        if off_axis_x is None:
            return f"main_{extra_label}.pdf"
        return f"{extra_label}_off{float(off_axis_x):.3f}.pdf"

    mocker.patch.object(ray_tracing_lst, "_generate_file_name", side_effect=_fake_name)

    # Provide two fake PSFImage instances.
    img0 = mocker.Mock()
    img1 = mocker.Mock()
    ray_tracing_lst.psf_images = {(0.0, 0.0): img0, (2.0, 0.0): img1}

    ray_tracing_lst._results = QTable(
        {
            "off axis angle": [0.0, 2.0],
            "psf_cm": [4.2, 4.3],
            "psf_deg": [0.1, 0.2],
            "eff_area": [100.0, 110.0],
            "eff_flen": [200.0, 210.0],
        }
    )

    ray_tracing_lst.plot(key="psf_cm", save=True, psf_diameter_cm=12.3)

    # Per-off-axis image export
    img0.plot_image.assert_called_once()
    img1.plot_image.assert_called_once()
    f0 = img0.plot_image.call_args.kwargs["file_name"]
    f1 = img1.plot_image.call_args.kwargs["file_name"]
    assert str(f0).endswith("image_psf_cm_off0.000.pdf")
    assert str(f1).endswith("image_psf_cm_off2.000.pdf")

    # Per-off-axis cumulative export
    img0.plot_cumulative.assert_called_once()
    img1.plot_cumulative.assert_called_once()
    c0 = img0.plot_cumulative.call_args.kwargs["file_name"]
    c1 = img1.plot_cumulative.call_args.kwargs["file_name"]
    assert str(c0).endswith("cumulative_psf_psf_cm_off0.000.pdf")
    assert str(c1).endswith("cumulative_psf_psf_cm_off2.000.pdf")
    assert img0.plot_cumulative.call_args.kwargs["psf_diameter_cm"] == pytest.approx(12.3)
    assert img1.plot_cumulative.call_args.kwargs["psf_diameter_cm"] == pytest.approx(12.3)


def test_load_offset_file_valid(
    tmp_test_directory, telescope_model_lst_mock, site_model_north, run_context
):
    offset_file = tmp_test_directory.join("offsets.ecsv")
    offset_content = """# %ECSV 1.0
# ---
# datatype:
# - {name: x, unit: deg, datatype: float64}
# - {name: y, unit: deg, datatype: float64}
x y
0.0 0.0
1.0 0.5
2.0 1.5
"""
    offset_file.write(offset_content)

    ray_tracing = RayTracing(
        telescope_model=telescope_model_lst_mock,
        site_model=site_model_north,
        label="test",
        zenith_angle=20 * u.deg,
        source_distance=10 * u.km,
        offset_file=offset_file,
        context=run_context,
    )

    assert len(ray_tracing.off_axis_angle) == 3
    assert ray_tracing.off_axis_angle[0] == (0.0, 0.0)
    assert ray_tracing.off_axis_angle[1] == (1.0, 0.5)
    assert ray_tracing.off_axis_angle[2] == (2.0, 1.5)


def test_load_offset_file_not_found(telescope_model_lst_mock, site_model_north, run_context):
    with pytest.raises(FileNotFoundError, match="Offset file not found"):
        RayTracing(
            telescope_model=telescope_model_lst_mock,
            site_model=site_model_north,
            label="test",
            zenith_angle=20 * u.deg,
            source_distance=10 * u.km,
            offset_file="/nonexistent/path/offsets.ecsv",
            context=run_context,
        )


def test_load_offset_file_missing_columns(
    tmp_test_directory, telescope_model_lst_mock, site_model_north, run_context
):
    offset_file = tmp_test_directory.join("offsets.ecsv")
    offset_content = """# %ECSV 1.0
# ---
# datatype:
# - {name: x, unit: deg, datatype: float64}
x
0.0
1.0
"""
    offset_file.write(offset_content)

    with pytest.raises(RuntimeError, match="Error reading offset file"):
        RayTracing(
            telescope_model=telescope_model_lst_mock,
            site_model=site_model_north,
            label="test",
            zenith_angle=20 * u.deg,
            source_distance=10 * u.km,
            offset_file=offset_file,
            context=run_context,
        )


def test_process_offset_angles_unknown_direction(
    telescope_model_lst_mock, site_model_north, caplog, run_context
):
    ray = RayTracing(
        telescope_model=telescope_model_lst_mock,
        site_model=site_model_north,
        label="test",
        zenith_angle=20 * u.deg,
        source_distance=10 * u.km,
        context=run_context,
    )
    with caplog.at_level(logging.WARNING):
        result = ray._process_offset_angles([1.0] * u.deg, ["N", "X"])
    assert len(result) == 1
    assert result[0] == (0.0, 1.0)
    assert "Unknown direction X" in caplog.text


@pytest.fixture
def backend(ray_tracing_lst, mocker):
    """A mocked backend returning one image that did not trace the telescope structure."""
    ray_tracing_lst._backend = mocker.Mock()
    ray_tracing_lst._backend.psf_image.return_value = ("psf_image", False)
    return ray_tracing_lst._backend


@pytest.mark.parametrize(
    ("test", "photon_count", "photons_per_instance", "instances"),
    [(False, None, 100000, 20), (True, None, 5000, 1), (False, 12345, 12345, 20)],
)
def test_simulate_requests(
    ray_tracing_lst, backend, test, photon_count, photons_per_instance, instances
):
    _set_context_args(
        ray_tracing_lst,
        {"number_of_photons": photon_count, "instrument_seed": 5, "simulation_seed": [7]},
    )
    ray_tracing_lst.off_axis_angle = [(0.0, 0.0), (1.0, -0.5)]

    ray_tracing_lst.simulate(test=test, force=True, compress_photons=False, max_workers=3)

    requests = backend.simulate.call_args.args[0]
    assert backend.simulate.call_args.kwargs == {
        "test": test,
        "force": True,
        "compress_photons": False,
        "max_workers": 3,
    }
    seeds = tuple(
        zip(
            random.instance_seeds(5, instances, max_seed=SIMTEL_MAX_SEED),
            random.instance_seeds(7, instances, max_seed=SIMTEL_MAX_SEED),
            strict=True,
        )
    )
    assert [(r.off_axis_x, r.off_axis_y) for r in requests] == [(0.0, 0.0), (1.0, -0.5)]
    for request in requests:
        assert request.source_distance_km == pytest.approx(10.0)
        assert request.photons_per_instance == photons_per_instance
        assert request.seeds == seeds
        assert request.mirror_number is None
        assert request.parameter_changes == {}


@pytest.mark.parametrize("photon_count", [0, -1, True, 1.5])
def test_simulate_rejects_invalid_photon_count(ray_tracing_lst, backend, photon_count):
    _set_context_args(ray_tracing_lst, {"number_of_photons": photon_count})
    with pytest.raises(ValueError, match="positive integer"):
        ray_tracing_lst.simulate()


def test_simulate_single_mirror_requests(ray_tracing_lst_single_mirror_mode, mocker):
    ray_tracing = ray_tracing_lst_single_mirror_mode
    backend = ray_tracing._backend = mocker.Mock()
    _set_context_args(ray_tracing, {"instrument_seed": 5, "simulation_seed": [7]})
    ray_tracing.telescope_model.get_single_mirror_list_file.return_value = Path("mirror.dat")

    ray_tracing.simulate()

    requests = backend.simulate.call_args.args[0]
    for request, (number, mirror_data) in zip(requests, ray_tracing.mirrors.items(), strict=True):
        assert request.mirror_number == number
        assert request.photons_per_instance == 100000
        # One instance; each mirror has its own seeds, derived from the configured ones.
        assert request.seeds == (
            tuple(
                random.seeds(fixed_seed=(seed, number + 1), max_seed=SIMTEL_MAX_SEED)
                for seed in (5, 7)
            ),
        )
        assert request.parameter_changes["mirror_list"] == "mirror.dat"
        assert request.parameter_changes["focal_length"] == pytest.approx(
            mirror_data["source_distance"] * 1.0e5
        )


@pytest.mark.parametrize("do_analyze", [True, False])
@pytest.mark.parametrize("keep_images", [True, False])
def test_process_off_axis_and_mirror(ray_tracing_lst, backend, mocker, do_analyze, keep_images):
    analyze_image = mocker.patch.object(ray_tracing_lst, "_analyze_image", return_value=(1.0, 2.0))
    ray_tracing_lst.psf_images = {}

    results = ray_tracing_lst._process_off_axis_and_mirror(
        [0.5, 0, 0, 0], do_analyze, False, 0.8, save_photons=True, keep_images=keep_images
    )

    request = backend.psf_image.call_args.args[0]
    assert request.key == (0.0, 0.0, None)
    assert backend.psf_image.call_args.args[1:] == (
        ray_tracing_lst.mirrors[0]["focal_length"],
        0.8,
    )
    assert backend.psf_image.call_args.kwargs == {"use_rx": False, "save_photons": True}
    assert len(results) == (1 if do_analyze else 0)
    assert analyze_image.called is do_analyze
    assert len(ray_tracing_lst.psf_images) == (1 if keep_images else 0)


@pytest.mark.parametrize("traced_structure", [False, True])
def test_traced_structure_replaces_telescope_transmission(
    ray_tracing_lst, backend, mocker, traced_structure
):
    """telescope_transmission applies unless the backend traced the telescope structure."""
    backend.psf_image.return_value = ("psf_image", traced_structure)
    analyze_image = mocker.patch.object(ray_tracing_lst, "_analyze_image", return_value=(1.0,))

    ray_tracing_lst._process_off_axis_and_mirror([0.5, 0, 0, 0], True, False, 0.8, False)

    assert analyze_image.call_args.args[-1] == pytest.approx(1.0 if traced_structure else 0.5)


def test_backend_is_selected_by_configuration(ray_tracing_lst, mocker):
    backend_class = mocker.patch("simtools.ray_tracing.ray_tracing.interface.get_backend")
    assert ray_tracing_lst._get_backend() is backend_class.return_value.return_value
    assert ray_tracing_lst._get_backend() is backend_class.return_value.return_value
    backend_class.return_value.assert_called_once_with(
        ray_tracing_lst.telescope_model,
        ray_tracing_lst.site_model,
        ray_tracing_lst.label,
        ray_tracing_lst.zenith_angle,
        False,
        context=ray_tracing_lst.context,
    )

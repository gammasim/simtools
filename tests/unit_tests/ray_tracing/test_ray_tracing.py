#!/usr/bin/python3

import copy
import logging
import shutil
from math import pi, tan
from pathlib import Path
from unittest.mock import call

import astropy.units as u
import pytest
from astropy.table import QTable

from simtools.ray_tracing.ray_tracing import INVALID_KEY_TO_PLOT, RayTracing


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
def ray_tracing_lst(telescope_model_lst_mock, site_model_north):
    """A RayTracing instance with results read in that were simulated before"""

    ray_tracing_lst = RayTracing(
        telescope_model=telescope_model_lst_mock,
        site_model=site_model_north,
        label="validate_optics",
        source_distance=10 * u.km,
        zenith_angle=20 * u.deg,
        off_axis_angle=[0] * u.deg,
    )

    output_directory = ray_tracing_lst.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        "tests/resources/ray_tracing_North_LSTN-01_d10.0km_za20.0deg_validate_optics.ecsv",
        output_directory.joinpath("results"),
    )
    shutil.copy(
        "tests/resources/ray_tracing_photons_North_LSTN-01_d10.0km_za20.0deg_off0.000"
        "deg_validate_optics.lis.gz",
        output_directory,
    )
    return ray_tracing_lst


@pytest.fixture
def ray_tracing_lst_single_mirror_mode(telescope_model_lst_mock, site_model_north):
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
    )


def test_ray_tracing_init(telescope_model_lst_mock, site_model_north, caplog):
    with caplog.at_level(logging.DEBUG):
        ray = RayTracing(
            telescope_model=telescope_model_lst_mock,
            site_model=site_model_north,
            label=telescope_model_lst_mock.label,
            zenith_angle=30 * u.deg,
            source_distance=10 * u.km,
            off_axis_angle=[0, 2] * u.deg,
        )

    assert ray.zenith_angle == pytest.approx(30)
    assert len(ray.off_axis_angle) == 2
    assert "Initializing RayTracing class" in caplog.text
    assert repr(ray) == f"RayTracing(label={telescope_model_lst_mock.label})\n"


def test_ray_tracing_single_mirror_mode(telescope_model_lst_mock, site_model_north, caplog):
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
        )

    assert ray.zenith_angle == pytest.approx(30)
    assert len(ray.off_axis_angle) == 2
    assert ray.single_mirror_mode
    assert "Single mirror mode is activated" in caplog.text
    assert len(ray.mirrors) == telescope_model_lst_mock.mirrors.number_of_mirrors


def test_ray_tracing_single_mirror_mode_mirror_numbers(
    telescope_model_lst_mock, site_model_north, mocker
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
    )

    assert list(ray.mirrors.keys()) == [1, 2, 3]

    mocker.patch.object(
        ray.telescope_model.mirrors, "get_single_mirror_parameters", return_value=(0, 0, 0, 0, 0)
    )
    mocker.patch.object(ray, "_get_mirror_panel_focal_length", return_value=0)
    with pytest.raises(ValueError, match=r"Focal length is invalid \(NaN or close to zero\)"):
        ray._initialize_single_mirror_mode([1, 2, 3])


def test_export_results(ray_tracing_lst, caplog, mocker):
    """
    Test the export_results method of the RayTracing class without results
    """

    ray = copy.deepcopy(ray_tracing_lst)
    ray.export_results()
    assert "No results to export" in caplog.text

    _rows = [
        (0.0, 4.256768651160611, 0.1, 100.0, 200.0),
        (2.0, 4.356768651160611, 0.2, 110.0, 210.0),
    ]
    ray._store_results(_rows)
    mock_write = mocker.patch("astropy.io.ascii.write")
    with caplog.at_level(logging.INFO):
        ray.export_results()
    assert "Exporting results" in caplog.text
    mock_write.assert_called_once_with(
        ray._results, ray._file_results, format="ecsv", overwrite=True
    )


def test_ray_tracing_no_images(ray_tracing_lst, caplog):
    """Test the images method of the RayTracing class with no images"""

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
    mock_read_results = mocker.patch.object(ray_tracing_lst, "_read_results")
    mock_store_results = mocker.patch.object(ray_tracing_lst, "_store_results")
    mock_export_results = mocker.patch.object(ray_tracing_lst, "export_results")
    mock_exists = mocker.patch("pathlib.Path.exists", return_value=True)
    mock_process = mocker.patch.object(ray_tracing_lst, "_process_off_axis_and_mirror")

    ray_tracing_lst.analyze(export=export, force=force)

    mock_exists.assert_called_once()
    mock_process.assert_called_once()

    assert mock_read_results.called == read_called
    assert mock_store_results.called == store_called
    assert mock_export_results.called == export_called


def test_process_off_axis_and_mirror(ray_tracing_lst, mocker):
    mock_generate_file_name = mocker.patch.object(
        ray_tracing_lst, "_generate_file_name", return_value="photons_file"
    )
    mock_create_psf_image = mocker.patch.object(
        ray_tracing_lst, "_create_psf_image", return_value="psf_image"
    )
    mock_analyze_image = mocker.patch.object(
        ray_tracing_lst,
        "_analyze_image",
        return_value=(0.5 * u.deg, 5.0 * u.cm, 10.0 * u.deg, 100.0 * u.m * u.m),
    )

    all_mirrors = {
        0: {
            "focal_length": 10.0,
        },
        1: {
            "focal_length": 20.0,
        },
        2: {
            "focal_length": 30.0,
        },
    }
    ray_tracing_lst.mirrors = all_mirrors
    ray_tracing_lst.single_mirror_mode = True
    tel_transmission_pars = ray_tracing_lst._get_telescope_transmission_params(True)
    do_analyze = True
    use_rx = False
    containment_fraction = 0.8

    results = ray_tracing_lst._process_off_axis_and_mirror(
        tel_transmission_pars,
        do_analyze,
        use_rx,
        containment_fraction,
    )

    assert len(results) == len(ray_tracing_lst.off_axis_angle) * len(all_mirrors)

    mock_generate_file_name.assert_called()
    mock_create_psf_image.assert_called()
    mock_analyze_image.assert_called()


def test_process_off_axis_and_mirror_no_analyze(ray_tracing_lst, mocker):
    mock_generate_file_name = mocker.patch.object(
        ray_tracing_lst, "_generate_file_name", return_value="photons_file"
    )
    mock_create_psf_image = mocker.patch.object(
        ray_tracing_lst, "_create_psf_image", return_value="psf_image"
    )
    mock_analyze_image = mocker.patch.object(ray_tracing_lst, "_analyze_image")

    tel_transmission_pars = ray_tracing_lst._get_telescope_transmission_params(True)
    do_analyze = False
    use_rx = False
    containment_fraction = 0.8

    results = ray_tracing_lst._process_off_axis_and_mirror(
        tel_transmission_pars,
        do_analyze,
        use_rx,
        containment_fraction,
    )

    assert len(results) == 0

    mock_generate_file_name.assert_called()
    mock_create_psf_image.assert_called()
    mock_analyze_image.assert_not_called()


def test_images_with_psf_images(ray_tracing_lst, mocker):
    mock_psf_image = mocker.Mock()
    ray_tracing_lst._psf_images = {0: mock_psf_image}

    images = ray_tracing_lst.images()

    assert images is not None
    assert len(images) == 1
    assert images[0] == mock_psf_image


def test_images_no_psf_images(ray_tracing_lst, caplog):
    ray_tracing_lst._psf_images = {}

    with caplog.at_level(logging.WARNING):
        images = ray_tracing_lst.images()

    assert images is None
    assert "No image found" in caplog.text


def test_store_results(ray_tracing_lst, ray_tracing_lst_single_mirror_mode, off_axis_string):
    """
    Test the _store_results method of the RayTracing class.
    """
    _rows = [
        (0.0, 4.256768651160611, 0.1, 100.0, 200.0),
        (2.0, 4.356768651160611, 0.2, 110.0, 210.0),
    ]
    ray_tracing_lst._store_results(_rows)

    assert isinstance(ray_tracing_lst._results, QTable)
    assert len(ray_tracing_lst._results) == len(_rows)
    assert ray_tracing_lst._results.colnames == [
        off_axis_string,
        "psf_cm",
        "psf_deg",
        "eff_area",
        "eff_flen",
    ]

    # single mirror mode
    _rows = [
        (0.0, 4.256768651160611, 0.1, 100.0, 200.0, 20),
        (2.0, 4.356768651160611, 0.2, 110.0, 210.0, 21),
    ]
    ray_tracing_lst_single_mirror_mode._store_results(_rows)
    assert len(ray_tracing_lst_single_mirror_mode._results) == len(_rows)


def test_store_results_single_mirror_mode(ray_tracing_lst_single_mirror_mode, off_axis_string):
    """
    Test the _store_results method of the RayTracing class with single mirror mode.
    """
    _rows = [
        (0.0, 4.256768651160611, 0.1, 100.0, 200.0, 1),
        (2.0, 4.356768651160611, 0.2, 110.0, 210.0, 2),
    ]
    ray_tracing_lst_single_mirror_mode.YLABEL = {
        "psf_cm": "Containment diameter (cm)",
        "psf_deg": "Containment diameter (deg)",
        "eff_area": "Effective area (m^2)",
        "eff_flen": "Effective focal length (cm)",
    }
    ray_tracing_lst_single_mirror_mode._store_results(_rows)

    assert isinstance(ray_tracing_lst_single_mirror_mode._results, QTable)
    assert len(ray_tracing_lst_single_mirror_mode._results) == len(_rows)
    assert ray_tracing_lst_single_mirror_mode._results.colnames == [
        off_axis_string,
        "psf_cm",
        "psf_deg",
        "eff_area",
        "eff_flen",
        "mirror_number",
    ]


def test_get_mirror_panel_focal_length_no_random(ray_tracing_lst, mocker):
    """
    Test without random focal length.
    """
    mock_get_parameter_value = mocker.patch.object(
        ray_tracing_lst.telescope_model, "get_parameter_value", return_value=10.0
    )
    ray_tracing_lst.use_random_focal_length = False

    focal_length = ray_tracing_lst._get_mirror_panel_focal_length()

    assert focal_length == pytest.approx(10.0)
    mock_get_parameter_value.assert_called_once_with("mirror_focal_length")


def test_get_mirror_panel_focal_length_with_random_normal(ray_tracing_lst, mocker):
    """
    Test with random focal length using normal distribution.
    """
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
    """
    Test with random focal length using uniform distribution.
    """
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


def test_ray_tracing_simulate(ray_tracing_lst, site_model_north, caplog, mocker):
    mock_simulator = mocker.patch("simtools.ray_tracing.ray_tracing.SimulatorRayTracing")
    mock_simulator_instance = mock_simulator.return_value
    mock_simulator_instance.run = mocker.Mock()
    mock_open = mocker.patch("builtins.open", mocker.mock_open())
    mock_gzip_open = mocker.patch("gzip.open", mocker.mock_open())
    mock_copyfileobj = mocker.patch("shutil.copyfileobj")
    mock_unlink = mocker.patch("pathlib.Path.unlink")

    with caplog.at_level(logging.INFO):
        ray_tracing_lst.simulate(test=True, force=True)

    assert "Simulating RayTracing for off_axis=0.0, mirror=0" in caplog.text
    mock_simulator.assert_called_once_with(
        telescope_model=ray_tracing_lst.telescope_model,
        site_model=site_model_north,
        label=ray_tracing_lst.label,
        test=True,
        config_data={
            "zenith_angle": ray_tracing_lst.zenith_angle,
            "off_axis_angle": 0.0,
            "source_distance": ray_tracing_lst.mirrors[0]["source_distance"],
            "single_mirror_mode": ray_tracing_lst.single_mirror_mode,
            "use_random_focal_length": ray_tracing_lst.use_random_focal_length,
            "mirror_numbers": 0,
        },
        force_simulate=True,
    )
    mock_simulator_instance.run.assert_called_once_with(test=True)

    mock_open.assert_called_once()
    mock_gzip_open.assert_called_once()
    mock_copyfileobj.assert_called_once()
    mock_unlink.assert_called_once()

    photons_file = ray_tracing_lst.output_directory.joinpath(
        ray_tracing_lst._generate_file_name(
            file_type="photons",
            suffix=".lis",
            off_axis_angle=0.0,
            mirror_number=0,
        )
    )
    assert photons_file.with_suffix(photons_file.suffix + ".gz").exists() is True
    assert not photons_file.exists()


def test_get_telescope_transmission_params_no_transmission(ray_tracing_lst):
    """
    Test _get_telescope_transmission_params with no_tel_transmission=True.
    """
    result = ray_tracing_lst._get_telescope_transmission_params(no_tel_transmission=True)
    assert result == [1, 0, 0, 0]


def test_get_telescope_transmission_params_with_transmission(ray_tracing_lst, mocker):
    """
    Test _get_telescope_transmission_params with no_tel_transmission=False.
    """
    mock_get_parameter_value = mocker.patch.object(
        ray_tracing_lst.telescope_model, "get_parameter_value", return_value=[0.9, 0.1, 0.05, 0.02]
    )
    result = ray_tracing_lst._get_telescope_transmission_params(no_tel_transmission=False)
    assert result == [0.9, 0.1, 0.05, 0.02]
    mock_get_parameter_value.assert_called_once_with("telescope_transmission")


def test_create_psf_image(ray_tracing_lst, mocker, test_photons_file):
    """
    Test the _create_psf_image method of the RayTracing class.
    """
    mock_psf_image = mocker.patch("simtools.ray_tracing.ray_tracing.PSFImage")
    mock_psf_image_instance = mock_psf_image.return_value
    mock_process_photon_list = mocker.patch.object(mock_psf_image_instance, "process_photon_list")

    focal_length = 10.0
    this_off_axis = 0.0
    containment_fraction = 0.8
    use_rx = False

    image = ray_tracing_lst._create_psf_image(
        photons_file=test_photons_file,
        focal_length=focal_length,
        this_off_axis=this_off_axis,
        containment_fraction=containment_fraction,
        use_rx=use_rx,
    )

    mock_psf_image.assert_called_once_with(
        focal_length=focal_length,
        containment_fraction=containment_fraction,
    )
    mock_process_photon_list.assert_called_once_with(test_photons_file, use_rx)
    assert ray_tracing_lst._psf_images[this_off_axis] == mock_psf_image_instance
    assert image == mock_psf_image_instance


def test_analyze_image(ray_tracing_lst, mocker):
    mock_image = mocker.Mock()
    mock_image.get_psf.side_effect = [5.0, 0.1]
    mock_image.get_effective_area.return_value = 100.0
    mock_image.centroid_x = 0.5

    this_off_axis = 2.0
    containment_fraction = 0.8
    tel_transmission = 0.9

    result = ray_tracing_lst._analyze_image(
        image=mock_image,
        this_off_axis=this_off_axis,
        containment_fraction=containment_fraction,
        tel_transmission=tel_transmission,
    )

    assert result == (
        2.0 * u.deg,
        5.0 * u.cm,
        0.1 * u.deg,
        100.0 * u.m * u.m,
        mock_image.centroid_x / tan(this_off_axis * pi / 180.0),
    )
    mock_image.get_psf.assert_has_calls(
        [call(containment_fraction, "cm"), call(containment_fraction, "deg")]
    )
    mock_image.get_effective_area.assert_called_once_with(tel_transmission)


def test_get_mean_std(ray_tracing_lst):
    ray_tracing = copy.deepcopy(ray_tracing_lst)
    _rows = [
        (0.0, 4.256768651160611, 0.1, 100.0, 200.0),
        (2.0, 4.356768651160611, 0.2, 110.0, 210.0),
    ]
    ray_tracing._store_results(_rows)
    mean_value = ray_tracing.get_mean(key="psf_cm")
    std_value = ray_tracing.get_std_dev(key="psf_cm")
    assert mean_value == pytest.approx(4.3, abs=1e-2)
    assert std_value == pytest.approx(0.05, abs=1e-2)

    with pytest.raises(KeyError, match=INVALID_KEY_TO_PLOT):
        ray_tracing.get_mean(key="abc")
    with pytest.raises(KeyError, match=INVALID_KEY_TO_PLOT):
        ray_tracing.get_std_dev(key="abc")


def test_read_results(ray_tracing_lst, mocker):
    """
    Test the _read_results method of the RayTracing class.
    """
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
    """
    Test the plot_histogram method of the RayTracing class with a valid key.
    """
    mock_gca = mocker.patch("matplotlib.pyplot.gca")
    mock_ax = mock_gca.return_value
    mock_hist = mocker.patch.object(mock_ax, "hist")

    ray_tracing_lst._results = QTable(
        {
            "psf_cm": [4.256768651160611, 4.356768651160611],
            "psf_deg": [0.1, 0.2],
            "eff_area": [100.0, 110.0],
            "eff_flen": [200.0, 210.0],
        }
    )

    ray_tracing_lst.plot_histogram(key="psf_cm", bins=10)

    mock_gca.assert_called_once()
    mock_hist.assert_called_once_with(ray_tracing_lst._results["psf_cm"], bins=10)


def test_plot_histogram_invalid_key(ray_tracing_lst):
    """
    Test the plot_histogram method of the RayTracing class with an invalid key.
    """
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


def test_plot_valid_key(ray_tracing_lst, mocker, off_axis_string):
    """
    Test the plot method of the RayTracing class with a valid key.
    """
    mock_visualize_plot_table = mocker.patch(
        "simtools.ray_tracing.ray_tracing.visualize.plot_table"
    )
    mock_plot = mock_visualize_plot_table.return_value
    mock_generate_file_name = mocker.patch.object(
        ray_tracing_lst, "_generate_file_name", return_value="plot_file.pdf"
    )
    mock_mkdir = mocker.patch("pathlib.Path.mkdir")
    mock_savefig = mocker.patch.object(mock_plot, "savefig")

    ray_tracing_lst._results = QTable(
        {
            off_axis_string: [0.0, 2.0],
            "psf_cm": [4.256768651160611, 4.356768651160611],
            "psf_deg": [0.1, 0.2],
            "eff_area": [100.0, 110.0],
            "eff_flen": [200.0, 210.0],
        }
    )

    ray_tracing_lst.plot(key="psf_cm", save=True)

    mock_visualize_plot_table.assert_called_once()
    mock_generate_file_name.assert_called_once_with(
        file_type="ray_tracing",
        suffix=".pdf",
        extra_label="psf_cm",
    )
    mock_mkdir.assert_called_once_with(exist_ok=True)
    mock_savefig.assert_called_once_with(
        ray_tracing_lst.output_directory.joinpath("figures").joinpath("plot_file.pdf")
    )


def test_plot_invalid_key(ray_tracing_lst, off_axis_string):
    """
    Test the plot method of the RayTracing class with an invalid key.
    """
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


def test_plot_save_writes_psf_images_and_cumulative(ray_tracing_lst, mocker, off_axis_string):
    """Cover the save=True branch that exports PSF images and cumulative plots."""

    mock_visualize_plot_table = mocker.patch(
        "simtools.ray_tracing.ray_tracing.visualize.plot_table"
    )
    mock_plot = mock_visualize_plot_table.return_value
    mocker.patch.object(mock_plot, "savefig")

    # Avoid real filesystem writes; only validate paths passed to methods.
    mocker.patch("pathlib.Path.mkdir")

    def _fake_name(*, file_type, suffix, off_axis_angle=None, mirror_number=None, extra_label=None):
        assert file_type == "ray_tracing"
        assert suffix == ".pdf"
        if off_axis_angle is None:
            return f"main_{extra_label}.pdf"
        return f"{extra_label}_off{float(off_axis_angle):.3f}.pdf"

    mocker.patch.object(ray_tracing_lst, "_generate_file_name", side_effect=_fake_name)

    # Provide two fake PSFImage instances.
    img0 = mocker.Mock()
    img1 = mocker.Mock()
    ray_tracing_lst._psf_images = {0.0: img0, 2.0: img1}

    ray_tracing_lst._results = QTable(
        {
            off_axis_string: [0.0, 2.0],
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

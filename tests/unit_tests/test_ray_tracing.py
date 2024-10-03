#!/usr/bin/python3

import copy
import logging
import shutil

import astropy.units as u
import pytest
from astropy.table import QTable

from simtools.ray_tracing import RayTracing
from simtools.utils import names


@pytest.fixture
def ray_tracing_lst(telescope_model_lst, simtel_path, io_handler):
    """A RayTracing instance with results read in that were simulated before"""

    ray_tracing_lst = RayTracing(
        telescope_model=telescope_model_lst,
        simtel_path=simtel_path,
        label="validate_optics",
        source_distance=10 * u.km,
        zenith_angle=20 * u.deg,
        off_axis_angle=[0] * u.deg,
    )

    output_directory = ray_tracing_lst.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        "tests/resources/ray-tracing-North-LSTN-01-d10.0km-za20.0deg_validate_optics.ecsv",
        output_directory.joinpath("results"),
    )
    shutil.copy(
        "tests/resources/photons-North-LSTN-01-d10.0km-za20.0deg-off0.000"
        "deg_validate_optics.lis.gz",
        output_directory,
    )
    return ray_tracing_lst


@pytest.fixture
def ray_tracing_lst_single_mirror_mode(telescope_model_lst, simtel_path, io_handler):
    return RayTracing(
        telescope_model=telescope_model_lst,
        simtel_path=simtel_path,
        label="validate_optics",
        source_distance=10 * u.km,
        zenith_angle=20 * u.deg,
        off_axis_angle=[0] * u.deg,
        single_mirror_mode=True,
        mirror_numbers=[0, 2],
    )


def test_ray_tracing_init(simtel_path, io_handler, telescope_model_mst, caplog):

    with caplog.at_level(logging.DEBUG):
        ray = RayTracing(
            telescope_model=telescope_model_mst,
            simtel_path=simtel_path,
            zenith_angle=30 * u.deg,
            source_distance=10 * u.km,
            off_axis_angle=[0, 2] * u.deg,
        )

    assert pytest.approx(ray.config.zenith_angle) == 30
    assert len(ray.config.off_axis_angle) == 2
    assert "Initializing RayTracing class" in caplog.text
    assert ray.simtel_path == simtel_path
    assert repr(ray) == f"RayTracing(label={telescope_model_mst.label})\n"


def test_ray_tracing_single_mirror_mode(simtel_path, io_handler, telescope_model_mst, caplog):
    telescope_model_mst.export_config_file()

    with caplog.at_level(logging.DEBUG):
        ray = RayTracing(
            telescope_model=telescope_model_mst,
            simtel_path=simtel_path,
            zenith_angle=30 * u.deg,
            source_distance=10 * u.km,
            off_axis_angle=[0, 2] * u.deg,
            single_mirror_mode=True,
            mirror_numbers="all",
        )

    assert pytest.approx(ray.config.zenith_angle) == 30
    assert len(ray.config.off_axis_angle) == 2
    assert ray.config.single_mirror_mode
    assert "Single mirror mode is activated" in caplog.text
    assert len(ray.config.mirror_numbers) == telescope_model_mst.mirrors.number_of_mirrors


def test_ray_tracing_single_mirror_mode_mirror_numbers(
    simtel_path, io_handler, telescope_model_mst
):
    ray = RayTracing(
        telescope_model=telescope_model_mst,
        simtel_path=simtel_path,
        source_distance=10 * u.km,
        zenith_angle=30 * u.deg,
        off_axis_angle=[0, 2] * u.deg,
        single_mirror_mode=True,
        mirror_numbers=[1, 2, 3],
    )

    assert ray.config.mirror_numbers == [1, 2, 3]


def test_ray_tracing_read_results(ray_tracing_lst):
    ray_tracing_lst.analyze(force=False)

    assert len(ray_tracing_lst._results) > 0
    assert ray_tracing_lst.get_mean("d80_cm") == pytest.approx(4.256768651160611, abs=1e-5)


def test_export_results(simtel_path, ray_tracing_lst, caplog):
    """
    Test the export_results method of the RayTracing class without results
    """

    ray = ray_tracing_lst
    ray.export_results()
    assert "" in caplog.text


def test_ray_tracing_plot(ray_tracing_lst, caplog):
    """
    Test the plot method of the RayTracing class with an invalid key and a valid key
    """

    ray_tracing_lst.analyze(force=False)
    # First test a wrong key
    with pytest.raises(KeyError):
        ray_tracing_lst.plot(key="invalid_key")
    assert "Invalid key" in caplog.text

    # Now test a valid key
    with caplog.at_level(logging.INFO):
        ray_tracing_lst.plot(key="d80_cm", save=True)
        assert "Saving fig in" in caplog.text
    plot_file_name = names.generate_file_name(
        file_type="ray-tracing",
        suffix=".pdf",
        extra_label="d80_cm",
        site=ray_tracing_lst.telescope_model.site,
        telescope_model_name=ray_tracing_lst.telescope_model.name,
        source_distance=ray_tracing_lst.config.source_distance,
        zenith_angle=ray_tracing_lst.config.zenith_angle,
        label=ray_tracing_lst.label,
    )
    plot_file = ray_tracing_lst.output_directory.joinpath("figures").joinpath(plot_file_name)
    assert plot_file.exists() is True


def test_ray_tracing_invalid_key(ray_tracing_lst, caplog):
    """
    Test the a few methods of the RayTracing class with an invalid key
    """

    invalid_key = "Invalid key"
    with pytest.raises(KeyError):
        ray_tracing_lst.plot_histogram(key="invalid_key")
    assert invalid_key in caplog.text

    with pytest.raises(KeyError):
        ray_tracing_lst.get_mean(key="invalid_key")
    assert invalid_key in caplog.text

    with pytest.raises(KeyError):
        ray_tracing_lst.get_std_dev(key="invalid_key")
    assert invalid_key in caplog.text


def test_ray_tracing_get_std_dev(ray_tracing_lst):
    """Test the get_std_dev method of the RayTracing class"""

    ray_tracing_lst.analyze(force=False)
    assert ray_tracing_lst.get_std_dev(key="d80_cm") == pytest.approx(0.8418404935128992, abs=1e-5)


def test_ray_tracing_no_images(ray_tracing_lst, caplog):
    """Test the images method of the RayTracing class with no images"""

    assert ray_tracing_lst.images() is None
    assert "No image found" in caplog.text


def test_ray_tracing_simulate(ray_tracing_lst, caplog, mocker):
    mock_simulator = mocker.patch("simtools.ray_tracing.SimulatorRayTracing")
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
        simtel_path=ray_tracing_lst.simtel_path,
        telescope_model=ray_tracing_lst.telescope_model,
        test=True,
        config_data=ray_tracing_lst.config._replace(
            off_axis_angle=0.0,
            mirror_numbers=0,
        ),
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
            mirror_number=None,
        )
    )
    assert photons_file.with_suffix(photons_file.suffix + ".gz").exists() is True
    assert not photons_file.exists()


def test_analyze_no_force(ray_tracing_lst, mocker):
    mock_read_results = mocker.patch.object(ray_tracing_lst, "_read_results")
    mock_store_results = mocker.patch.object(ray_tracing_lst, "_store_results")
    mock_export_results = mocker.patch.object(ray_tracing_lst, "export_results")
    mock_exists = mocker.patch("pathlib.Path.exists", return_value=True)

    ray_tracing_lst.analyze(export=True, force=False)

    mock_exists.assert_called_once()
    mock_read_results.assert_called_once()
    mock_store_results.assert_not_called()
    mock_export_results.assert_called_once()


def test_analyze_force(ray_tracing_lst, mocker):
    mock_read_results = mocker.patch.object(ray_tracing_lst, "_read_results")
    mock_store_results = mocker.patch.object(ray_tracing_lst, "_store_results")
    mock_export_results = mocker.patch.object(ray_tracing_lst, "export_results")
    mock_process_off_axis_and_mirror = mocker.patch.object(
        ray_tracing_lst, "_process_off_axis_and_mirror"
    )
    mock_exists = mocker.patch("pathlib.Path.exists", return_value=True)

    ray_tracing_lst.analyze(export=True, force=True)

    mock_exists.assert_called_once()
    mock_process_off_axis_and_mirror.assert_called_once()
    mock_read_results.assert_not_called()
    mock_store_results.assert_called_once()
    mock_export_results.assert_called_once()


def test_analyze_no_export(ray_tracing_lst, mocker):
    mock_read_results = mocker.patch.object(ray_tracing_lst, "_read_results")
    mock_store_results = mocker.patch.object(ray_tracing_lst, "_store_results")
    mock_export_results = mocker.patch.object(ray_tracing_lst, "export_results")
    mock_exists = mocker.patch("pathlib.Path.exists", return_value=True)

    ray_tracing_lst.analyze(export=False, force=False)

    mock_exists.assert_called_once()
    mock_read_results.assert_called_once()
    mock_store_results.assert_not_called()
    mock_export_results.assert_not_called()


def test_process_off_axis_and_mirror(ray_tracing_lst, mocker):
    mock_generate_file_name = mocker.patch.object(
        ray_tracing_lst, "_generate_file_name", return_value="photons_file"
    )
    mock_compute_telescope_transmission = mocker.patch(
        "simtools.ray_tracing.compute_telescope_transmission", return_value=0.9
    )
    mock_create_psf_image = mocker.patch.object(
        ray_tracing_lst, "_create_psf_image", return_value="psf_image"
    )
    mock_analyze_image = mocker.patch.object(
        ray_tracing_lst, "_analyze_image", return_value=("result",)
    )

    all_mirrors = [0, 1, 2]
    focal_length = 10.0
    tel_transmission_pars = [0.9]
    cm_to_deg = 0.1
    do_analyze = True
    use_rx = False
    containment_fraction = 0.8

    results = ray_tracing_lst._process_off_axis_and_mirror(
        all_mirrors,
        focal_length,
        tel_transmission_pars,
        cm_to_deg,
        do_analyze,
        use_rx,
        containment_fraction,
    )

    assert len(results) == len(ray_tracing_lst.config.off_axis_angle) * len(all_mirrors)
    for result in results:
        assert result == ("result",)

    mock_generate_file_name.assert_called()
    mock_compute_telescope_transmission.assert_called()
    mock_create_psf_image.assert_called()
    mock_analyze_image.assert_called()


def test_process_off_axis_and_mirror_no_analyze(ray_tracing_lst, mocker):
    mock_generate_file_name = mocker.patch.object(
        ray_tracing_lst, "_generate_file_name", return_value="photons_file"
    )
    mock_compute_telescope_transmission = mocker.patch(
        "simtools.ray_tracing.compute_telescope_transmission", return_value=0.9
    )
    mock_create_psf_image = mocker.patch.object(
        ray_tracing_lst, "_create_psf_image", return_value="psf_image"
    )
    mock_analyze_image = mocker.patch.object(ray_tracing_lst, "_analyze_image")

    all_mirrors = [0, 1, 2]
    focal_length = 10.0
    tel_transmission_pars = [0.9]
    cm_to_deg = 0.1
    do_analyze = False
    use_rx = False
    containment_fraction = 0.8

    results = ray_tracing_lst._process_off_axis_and_mirror(
        all_mirrors,
        focal_length,
        tel_transmission_pars,
        cm_to_deg,
        do_analyze,
        use_rx,
        containment_fraction,
    )

    assert len(results) == 0

    mock_generate_file_name.assert_called()
    mock_compute_telescope_transmission.assert_called()
    mock_create_psf_image.assert_called()
    mock_analyze_image.assert_not_called()


def test_get_photons_file(ray_tracing_lst, mocker):
    ray_tracing = copy.deepcopy(ray_tracing_lst)

    this_off_axis = 0.0
    this_mirror = 1

    photons_file = ray_tracing._get_photons_file(this_off_axis, this_mirror)
    expected_file = ray_tracing.output_directory.joinpath(
        "photons-North-LSTN-01-d10.0km-za20.0deg-off0.000deg_validate_optics.lis.gz"
    )

    assert photons_file == expected_file


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


def test_plot_histogram_valid_key(ray_tracing_lst, mocker):
    ray_tracing_lst.analyze(force=False)
    mock_gca = mocker.patch("matplotlib.pyplot.gca")
    mock_hist = mocker.patch.object(mock_gca.return_value, "hist")

    ray_tracing_lst.plot_histogram(key="d80_cm")

    mock_gca.assert_called_once()
    mock_hist.assert_called_once()


def test_plot_histogram_invalid_key(ray_tracing_lst, caplog):
    invalid_key = "invalid_key"
    with pytest.raises(KeyError):
        ray_tracing_lst.plot_histogram(key=invalid_key)
    assert "Invalid key" in caplog.text


def test_store_results(ray_tracing_lst):
    """
    Test the _store_results method of the RayTracing class.
    """
    _rows = [
        (0.0, 4.256768651160611, 0.1, 100.0, 200.0),
        (2.0, 4.356768651160611, 0.2, 110.0, 210.0),
    ]
    ray_tracing_lst.YLABEL = {
        "d80_cm": "Containment diameter (cm)",
        "d80_deg": "Containment diameter (deg)",
        "eff_area": "Effective area (m^2)",
        "eff_flen": "Effective focal length (cm)",
    }

    ray_tracing_lst._store_results(_rows)

    assert isinstance(ray_tracing_lst._results, QTable)
    assert len(ray_tracing_lst._results) == len(_rows)
    assert ray_tracing_lst._results.colnames == [
        "Off-axis angle",
        "d80_cm",
        "d80_deg",
        "eff_area",
        "eff_flen",
    ]


def test_store_results_single_mirror_mode(ray_tracing_lst_single_mirror_mode):
    """
    Test the _store_results method of the RayTracing class with single mirror mode.
    """
    _rows = [
        (0.0, 4.256768651160611, 0.1, 100.0, 200.0, 1),
        (2.0, 4.356768651160611, 0.2, 110.0, 210.0, 2),
    ]
    ray_tracing_lst_single_mirror_mode.YLABEL = {
        "d80_cm": "Containment diameter (cm)",
        "d80_deg": "Containment diameter (deg)",
        "eff_area": "Effective area (m^2)",
        "eff_flen": "Effective focal length (cm)",
    }
    ray_tracing_lst_single_mirror_mode._store_results(_rows)

    assert isinstance(ray_tracing_lst_single_mirror_mode._results, QTable)
    assert len(ray_tracing_lst_single_mirror_mode._results) == len(_rows)
    assert ray_tracing_lst_single_mirror_mode._results.colnames == [
        "Off-axis angle",
        "d80_cm",
        "d80_deg",
        "eff_area",
        "eff_flen",
        "mirror_number",
    ]

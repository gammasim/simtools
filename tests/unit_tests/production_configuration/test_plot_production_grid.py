"""Tests for production grid plotting in production_configuration."""

from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.table import Table
from astropy.time import Time
from astropy.utils import iers

from simtools.production_configuration.plot_production_grid import (
    DEFAULT_OUTPUT_FILE_STEM,
    ProductionGridPlotter,
)

pytestmark = pytest.mark.filterwarnings("ignore::astropy.utils.iers.IERSWarning")


@pytest.fixture(autouse=True, scope="module")
def disable_iers_auto_download():
    """Disable IERS auto-download during tests to avoid network dependency."""
    previous_auto_download = iers.conf.auto_download
    iers.conf.auto_download = False
    try:
        yield
    finally:
        iers.conf.auto_download = previous_auto_download


SITE_LOCATION_LAT = 28.76
SITE_LOCATION_LON = -17.89
SITE_LOCATION_HEIGHT = 2200.0


def _write_grid_file(tmp_test_directory, file_name, grid_points):
    """Write grid points to a temporary ECSV file."""
    file_path = Path(tmp_test_directory) / file_name
    rows = []
    for point in grid_points:
        row = {}
        for key, value in point.items():
            if isinstance(value, dict) and "value" in value:
                row[key] = value["value"]
            else:
                row[key] = value
        rows.append(row)

    table = Table(rows=rows)
    if "azimuth" in table.colnames:
        table["azimuth"].unit = "deg"
    if "zenith_angle" in table.colnames:
        table["zenith_angle"].unit = "deg"
    if "ra" in table.colnames:
        table["ra"].unit = "deg"
    if "dec" in table.colnames:
        table["dec"].unit = "deg"
    if "nsb_level" in table.colnames:
        table["nsb_level"].unit = "MHz"
    if "offset" in table.colnames:
        table["offset"].unit = "deg"
    table.meta["observing_time_utc"] = "2025-01-01T00:00:00.000"
    table.write(file_path, format="ascii.ecsv", overwrite=True)
    return file_path


def _build_radec_mesh_grid_points(location, observation_time):
    """Build a small RA/Dec mesh around local sidereal time."""
    lst = observation_time.sidereal_time("apparent", longitude=location.lon).deg
    ra_values = [(lst - 5.0) % 360.0, (lst + 5.0) % 360.0]
    dec_values = [20.0, 30.0]

    return [
        {"ra": {"value": ra_value, "unit": "deg"}, "dec": {"value": dec_value, "unit": "deg"}}
        for dec_value in dec_values
        for ra_value in ra_values
    ]


def _create_plotter(grid_file, observation_time, output_path):
    """Create a plotter with the standard test site parameters."""
    return ProductionGridPlotter(
        grid_points_file=grid_file,
        site_location_lat=SITE_LOCATION_LAT,
        site_location_lon=SITE_LOCATION_LON,
        site_location_height=SITE_LOCATION_HEIGHT,
        observation_time=observation_time,
        output_path=output_path,
    )


def test_normalize_altaz_point_creates_radec_coordinates(tmp_test_directory):
    """Test that native Alt/Az points are converted to RA/Dec for the equatorial panel."""
    grid_file = _write_grid_file(
        tmp_test_directory,
        "grid_altaz.ecsv",
        [
            {
                "azimuth": {"value": 180.0, "unit": "deg"},
                "zenith_angle": {"value": 20.0, "unit": "deg"},
                "nsb_level": {"value": 0.0, "unit": "MHz"},
            }
        ],
    )

    plotter = _create_plotter(
        grid_file=grid_file,
        observation_time="2025-01-01 00:00:00",
        output_path=Path(tmp_test_directory) / "output",
    )

    normalized_points = plotter.normalize_grid_points()

    assert len(normalized_points) == 1
    assert normalized_points[0]["native_frame"] == "altaz"
    assert normalized_points[0]["visible_in_altaz"] is True
    assert normalized_points[0]["azimuth"] == pytest.approx(180.0)
    assert normalized_points[0]["zenith"] == pytest.approx(20.0)
    assert normalized_points[0]["ra"] is not None
    assert normalized_points[0]["dec"] is not None


def test_normalize_radec_point_projects_to_altaz(tmp_test_directory):
    """Test that native RA/Dec points are converted to Alt/Az for the local panel."""
    location = EarthLocation(
        lat=SITE_LOCATION_LAT * u.deg,
        lon=SITE_LOCATION_LON * u.deg,
        height=SITE_LOCATION_HEIGHT * u.m,
    )
    observation_time = Time("2025-01-01 00:00:00")
    source_altaz = SkyCoord(
        AltAz(
            alt=65.0 * u.deg,
            az=210.0 * u.deg,
            obstime=observation_time,
            location=location,
        )
    )
    source_radec = source_altaz.icrs

    grid_file = _write_grid_file(
        tmp_test_directory,
        "grid_radec.ecsv",
        [
            {
                "ra": {"value": source_radec.ra.deg, "unit": "deg"},
                "dec": {"value": source_radec.dec.deg, "unit": "deg"},
            }
        ],
    )

    plotter = _create_plotter(
        grid_file=grid_file,
        observation_time=str(observation_time.value),
        output_path=Path(tmp_test_directory) / "output",
    )

    normalized_points = plotter.normalize_grid_points()

    assert len(normalized_points) == 1
    assert normalized_points[0]["native_frame"] == "radec"
    assert normalized_points[0]["visible_in_altaz"] is True
    assert normalized_points[0]["azimuth"] == pytest.approx(210.0, abs=0.05)
    assert normalized_points[0]["zenith"] == pytest.approx(25.0, abs=0.05)
    assert normalized_points[0]["ra"] == pytest.approx(source_radec.ra.deg, abs=0.05)
    assert normalized_points[0]["dec"] == pytest.approx(source_radec.dec.deg, abs=0.05)


def test_infer_radec_grid_tracks_from_native_points(tmp_test_directory):
    """Infer RA and Dec grid tracks from a native RA/Dec mesh."""
    location = EarthLocation(
        lat=SITE_LOCATION_LAT * u.deg,
        lon=SITE_LOCATION_LON * u.deg,
        height=SITE_LOCATION_HEIGHT * u.m,
    )
    observation_time = Time("2025-01-01 00:00:00")
    grid_points = _build_radec_mesh_grid_points(location, observation_time)

    grid_file = _write_grid_file(tmp_test_directory, "grid_radec_mesh.ecsv", grid_points)
    plotter = _create_plotter(
        grid_file=grid_file,
        observation_time=str(observation_time.value),
        output_path=Path(tmp_test_directory) / "output",
    )

    normalized_points = plotter.normalize_grid_points()
    track_groups = plotter.infer_radec_grid_tracks(normalized_points)

    assert len(track_groups["declination_tracks"]) == 2
    assert len(track_groups["right_ascension_tracks"]) == 2
    assert all(len(group) == 2 for group in track_groups["declination_tracks"])
    assert all(len(group) == 2 for group in track_groups["right_ascension_tracks"])


def test_plot_sky_projection_creates_outputs(tmp_test_directory):
    """Test that the sky projection plot is written to disk."""
    grid_file = _write_grid_file(
        tmp_test_directory,
        "grid_wrapped.ecsv",
        [
            {
                "azimuth": {"value": 310.0, "unit": "deg"},
                "zenith_angle": {"value": 30.0, "unit": "deg"},
                "nsb_level": {"value": 4.0, "unit": "MHz"},
                "offset": {"value": 0.0, "unit": "deg"},
            }
        ],
    )
    output_path = Path(tmp_test_directory) / "output"

    plotter = _create_plotter(
        grid_file=grid_file,
        observation_time="2025-01-01 00:00:00",
        output_path=output_path,
    )

    plotter.plot_sky_projection(plot_ra_dec_tracks=True, dec_values=[20.0, 30.0])

    assert (output_path / f"{DEFAULT_OUTPUT_FILE_STEM}.png").exists()


def test_plot_sky_projection_infers_radec_grid_tracks(tmp_test_directory):
    """Plot inferred RA/Dec grid tracks."""
    location = EarthLocation(
        lat=SITE_LOCATION_LAT * u.deg,
        lon=SITE_LOCATION_LON * u.deg,
        height=SITE_LOCATION_HEIGHT * u.m,
    )
    observation_time = Time("2025-01-01 00:00:00")
    grid_points = _build_radec_mesh_grid_points(location, observation_time)

    grid_file = _write_grid_file(tmp_test_directory, "grid_radec_tracks.ecsv", grid_points)
    output_path = Path(tmp_test_directory) / "output"
    plotter = _create_plotter(
        grid_file=grid_file,
        observation_time=str(observation_time.value),
        output_path=output_path,
    )

    plotter.plot_sky_projection(plot_ra_dec_tracks=True)

    assert (output_path / f"{DEFAULT_OUTPUT_FILE_STEM}.png").exists()


def test_load_grid_points_from_ecsv(tmp_test_directory):
    """Test that ECSV grid-point files are loaded and normalized."""
    grid_file = Path(tmp_test_directory) / "grid_points.ecsv"
    table = Table(rows=[{"azimuth": 180.0, "zenith_angle": 20.0, "nsb_level": 1.0}])
    table["azimuth"].unit = "deg"
    table["zenith_angle"].unit = "deg"
    table["nsb_level"].unit = "MHz"
    table.meta["observing_time_utc"] = "2025-01-01T00:00:00.000"
    table.write(grid_file, format="ascii.ecsv", overwrite=True)

    plotter = _create_plotter(
        grid_file=grid_file,
        observation_time=None,
        output_path=Path(tmp_test_directory) / "output",
    )

    normalized_points = plotter.normalize_grid_points()
    assert len(normalized_points) == 1
    assert normalized_points[0]["native_frame"] == "altaz"


def test_load_grid_points_file_not_found(tmp_test_directory):
    """Raise when the grid points file does not exist."""
    missing_file = Path(tmp_test_directory) / "does_not_exist.ecsv"

    with pytest.raises(FileNotFoundError, match="Grid points file not found"):
        _create_plotter(
            grid_file=missing_file,
            observation_time="2025-01-01 00:00:00",
            output_path=Path(tmp_test_directory) / "output",
        )


def test_load_grid_points_wrong_suffix(tmp_test_directory):
    """Raise when the grid points file is not ECSV."""
    wrong_file = Path(tmp_test_directory) / "grid_points.txt"
    wrong_file.write_text("dummy", encoding="utf-8")

    with pytest.raises(ValueError, match="must be ECSV"):
        _create_plotter(
            grid_file=wrong_file,
            observation_time="2025-01-01 00:00:00",
            output_path=Path(tmp_test_directory) / "output",
        )


def test_extract_quantity_value_dict_branches():
    """Cover value/lower/None extraction paths."""
    point_with_value = {"x": {"value": 12.3, "unit": "deg"}}
    assert ProductionGridPlotter._extract_quantity_value(point_with_value, "x") == pytest.approx(
        12.3
    )

    point_with_lower = {"x": {"lower": {"value": 7.5, "unit": "deg"}}}
    assert ProductionGridPlotter._extract_quantity_value(point_with_lower, "x") == pytest.approx(
        7.5
    )

    point_without_value = {"x": {"unit": "deg"}}
    assert ProductionGridPlotter._extract_quantity_value(point_without_value, "x") is None


def test_configure_radec_axis_expands_flat_ranges(tmp_test_directory):
    """Expand axis limits by +/-5 deg when computed min and max are equal."""
    grid_file = _write_grid_file(
        tmp_test_directory,
        "grid_single_radec.ecsv",
        [{"ra": {"value": 40.0, "unit": "deg"}, "dec": {"value": 20.0, "unit": "deg"}}],
    )
    plotter = _create_plotter(
        grid_file=grid_file,
        observation_time="2025-01-01 00:00:00",
        output_path=Path(tmp_test_directory) / "output",
    )

    figure, axis = plt.subplots()
    try:
        plotter._configure_radec_axis(axis, [{"ra": 40.0, "dec": 20.0}])
        xlim = axis.get_xlim()
        ylim = axis.get_ylim()
        assert xlim == pytest.approx((45.0, 35.0))
        assert ylim == pytest.approx((15.0, 25.0))
    finally:
        plt.close(figure)


def test_plot_frame_points_logs_no_valid_points(tmp_test_directory, caplog):
    """Log warning and return zero when no points can be plotted."""
    grid_file = _write_grid_file(tmp_test_directory, "grid_empty.ecsv", [])
    plotter = _create_plotter(
        grid_file=grid_file,
        observation_time="2025-01-01 00:00:00",
        output_path=Path(tmp_test_directory) / "output",
    )

    figure, axis = plt.subplots()
    try:
        with caplog.at_level("WARNING"):
            plotted = plotter._plot_frame_points(
                axis=axis,
                plot_points=[],
                primary_frame="altaz",
                secondary_frame="radec",
                primary_label="A",
                secondary_label="B",
                primary_color="tab:blue",
                secondary_color="tab:orange",
                x_key="azimuth",
                y_key="zenith",
                panel_name="Alt/Az",
            )
        assert plotted == 0
        assert "No valid grid points found for Alt/Az plotting" in caplog.text
    finally:
        plt.close(figure)


def test_plot_altaz_points_logs_hidden_radec_points(tmp_test_directory, caplog):
    """Log info when RA/Dec points are below horizon and skipped in Alt/Az panel."""
    grid_file = _write_grid_file(tmp_test_directory, "grid_empty_altaz.ecsv", [])
    plotter = _create_plotter(
        grid_file=grid_file,
        observation_time="2025-01-01 00:00:00",
        output_path=Path(tmp_test_directory) / "output",
    )

    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1, projection="polar")
    try:
        plot_points = [
            {
                "native_frame": "radec",
                "azimuth": None,
                "zenith": None,
                "ra": 10.0,
                "dec": -80.0,
                "visible_in_altaz": False,
            }
        ]
        with caplog.at_level("INFO"):
            plotter._plot_altaz_points(axis, plot_points)
        assert "Skipping 1 RA/Dec points below the horizon in Alt/Az panel" in caplog.text
    finally:
        plt.close(figure)


def test_plot_inferred_radec_grid_logs_no_tracks(tmp_test_directory, caplog):
    """Log info when no inferred RA/Dec tracks can be plotted."""
    grid_file = _write_grid_file(tmp_test_directory, "grid_empty_tracks.ecsv", [])
    plotter = _create_plotter(
        grid_file=grid_file,
        observation_time="2025-01-01 00:00:00",
        output_path=Path(tmp_test_directory) / "output",
    )

    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1, projection="polar")
    try:
        with caplog.at_level("INFO"):
            plotted = plotter._plot_inferred_radec_grid(axis, plot_points=[])
        assert plotted == 0
        assert "No inferred RA/Dec grid tracks available for plotting" in caplog.text
    finally:
        plt.close(figure)


def test_iers_disabled_with_env_plotter(monkeypatch, tmp_test_directory):
    from simtools.application_control import _configure_iers_from_env

    iers.conf.auto_download = True
    iers.conf.auto_max_age = 30

    monkeypatch.setenv("SIMTOOLS_OFFLINE_IERS", "1")

    _configure_iers_from_env()

    grid_file = _write_grid_file(
        tmp_test_directory,
        "grid.ecsv",
        [{"azimuth": 0.0, "zenith_angle": 0.0}],
    )

    _create_plotter(
        grid_file=grid_file,
        observation_time="2025-01-01 00:00:00",
        output_path=Path(tmp_test_directory) / "output",
    )

    assert iers.conf.auto_download is False


def test_iers_not_modified_without_env_plotter(monkeypatch, tmp_test_directory):
    from simtools.application_control import _configure_iers_from_env

    iers.conf.auto_download = True
    iers.conf.auto_max_age = 30

    monkeypatch.delenv("SIMTOOLS_OFFLINE_IERS", raising=False)

    _configure_iers_from_env()

    grid_file = _write_grid_file(
        tmp_test_directory,
        "grid.ecsv",
        [{"azimuth": 0.0, "zenith_angle": 0.0}],
    )

    _create_plotter(
        grid_file=grid_file,
        observation_time="2025-01-01 00:00:00",
        output_path=Path(tmp_test_directory) / "output",
    )

    assert iers.conf.auto_download is True

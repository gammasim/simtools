from unittest.mock import Mock, patch

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.tests.helper import assert_quantity_allclose
from astropy.utils import iers

from simtools.production_configuration.grid import (
    MAX_SHOWERS_PER_RUN_ARGUMENT,
    NUMBER_OF_RUNS_ARGUMENT,
    TOTAL_SHOWERS_ARGUMENT,
    ProductionGridEngine,
    _build_rows_for_point,
    _generate_observation_grids_per_layout,
    _generate_observation_points_from_axes,
    _parse_axis_spec,
    _resolve_shower_params,
    _RunAllocationConfig,
    build_axes_dict_from_cli_args,
    build_job_grid_metadata,
    build_observing_location,
    build_production_grid_engine,
    build_simulation_jobs,
    calculate_log_energy_midpoint,
    calculate_scaled_showers_per_run,
    calculate_zenith_scaled_showers_per_run,
    ceil_with_tolerance,
    directed_circular_span_degrees,
    generate_job_grid,
    normalize_energy_ranges,
    renumber_job_rows,
    scale_energy_max_for_zenith_angle,
)
from simtools.production_configuration.jobs_io import build_job_grid_summary

ZENITH_ANGLE_KEY = "zenith_angle"
COSINE_ZENITH_SCALING = "cosine_zenith"

DEFAULT_OBSERVING_LOCATION = EarthLocation(
    lat=28.76 * u.deg,
    lon=-17.89 * u.deg,
    height=2200 * u.m,
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


def _make_engine(axes=None, **kwargs):
    """Build a production-grid engine with default wrapped axes input."""
    return ProductionGridEngine(axes={"axes": axes or {}}, **kwargs)


def _adaptive_hadec_axes(local_zenith_range=None, local_azimuth_range=None):
    """Return baseline HA/Dec adaptive-density axes with optional local constraints."""
    axes = {
        "ha": {
            "range": [0, 360],
            "binning": 36,
            "units": "deg",
            "direction_grid_density": 0.25,
        },
        "dec": {"range": [-40, 80], "binning": 10, "units": "deg"},
        "offset": {"range": [0, 10], "binning": 2, "units": "deg"},
    }
    if local_zenith_range is not None:
        axes["ha"]["local_zenith_range"] = local_zenith_range
    if local_azimuth_range is not None:
        axes["ha"]["local_azimuth_range"] = local_azimuth_range
    return axes


def _single_bin_horizontal_axes(azimuth_range=None):
    """Return single-bin horizontal axes used for interpolation-limit tests."""
    az_range = [0, 0] if azimuth_range is None else azimuth_range
    return {
        "zenith_angle": {"range": [20, 20], "binning": 1, "units": "deg"},
        "azimuth": {"range": az_range, "binning": 1 if az_range == [0, 0] else 3, "units": "deg"},
    }


def _base_simulation_jobs_args():
    return {
        "site": "North",
        "primary": ["gamma"],
        "azimuth_angle": [180 * u.deg],
        ZENITH_ANGLE_KEY: [20 * u.deg],
        "model_version": ["6.3.0"],
        "corsika_le_interaction": "urqmd",
        "corsika_he_interaction": "epos",
        "energy_range": [(30 * u.GeV, 100 * u.GeV)],
        "core_scatter": [10, 200 * u.m],
        "view_cone": [0 * u.deg, 5 * u.deg],
        "showers_per_run": 5,
        NUMBER_OF_RUNS_ARGUMENT: 1,
        "run_number": 11,
        "array_layout_name": "alpha",
    }


def _observation_grid_return(point):
    return ({"6.3.0": [point]}, {"6.3.0": "alpha"})


def _two_model_versions_grid_axes():
    return {
        "model_version": ["6.3.0", "7.0.0"],
        "azimuth_angle": [0 * u.deg],
        ZENITH_ANGLE_KEY: [20 * u.deg],
    }


def _horizontal_shared_axis_args(layout_by_version):
    return {
        "site": "North",
        "axis": [
            ["azimuth", "310", "deg", "20", "deg", "3"],
            ["zenith", "20", "deg", "40", "deg", "2"],
            ["offset", "0", "deg", "10", "deg", "2"],
        ],
        "array_layout_name": {"by_version": layout_by_version},
    }


def test_build_axes_dict_from_cli_args_derives_horizontal_binning_from_density():
    axes = build_axes_dict_from_cli_args(
        {
            "direction_grid_density": 1.0,
            "axis": [
                ["azimuth", "310", "deg", "20", "deg", "3", "linear"],
                ["zenith", "30", "deg", "40", "deg", "2"],
                ["offset", "0", "deg", "10", "deg", "2"],
            ],
        }
    )

    assert axes["azimuth"]["binning"] == 41
    assert axes[ZENITH_ANGLE_KEY]["binning"] == 10


def test_build_axes_dict_from_cli_args_accepts_density_with_unit():
    axes = build_axes_dict_from_cli_args(
        {
            "direction_grid_density": "1.0 1/deg^2",
            "axis": [
                ["azimuth", "310", "deg", "20", "deg", "3", "linear"],
                ["zenith", "30", "deg", "40", "deg", "2"],
                ["offset", "0", "deg", "10", "deg", "2"],
            ],
        }
    )

    assert axes["azimuth"]["direction_grid_density"] == pytest.approx(1.0)


def test_build_axes_dict_from_cli_args_derives_hadec_binning_from_density():
    axes = build_axes_dict_from_cli_args(
        {
            "direction_grid_density": 1.0,
            "axis": [
                ["ha", "0", "deg", "360", "deg", "36", "linear"],
                ["dec", "-90", "deg", "90", "deg", "18", "linear"],
                ["offset", "0", "deg", "10", "deg", "2", "linear"],
            ],
        }
    )

    assert axes["ha"]["binning"] == 230
    assert axes["dec"]["binning"] == 180
    assert axes["ha"]["direction_grid_density"] == pytest.approx(1.0)


def test_build_axes_dict_from_cli_args_keeps_horizontal_constraints_in_hadec_mode():
    axes = build_axes_dict_from_cli_args(
        {
            "direction_grid_density": 0.25,
            "local_zenith_range": ["0", "deg", "70", "deg"],
            "local_azimuth_range": ["300", "deg", "60", "deg"],
            "axis": [
                ["ha", "0", "deg", "360", "deg", "36", "linear"],
                ["dec", "-40", "deg", "80", "deg", "10", "linear"],
                ["offset", "0", "deg", "10", "deg", "2", "linear"],
            ],
        }
    )

    assert "ha" in axes
    assert "dec" in axes
    assert ZENITH_ANGLE_KEY not in axes
    assert "azimuth" not in axes
    assert axes["ha"]["local_zenith_range"] == pytest.approx([0.0, 70.0])
    assert axes["ha"]["local_azimuth_range"] == pytest.approx([300.0, 60.0])


def test_build_axes_dict_from_cli_args_rejects_non_positive_density():
    with pytest.raises(ValueError, match="direction_grid_density must be strictly positive"):
        build_axes_dict_from_cli_args(
            {
                "direction_grid_density": 0,
                "axis": [
                    ["azimuth", "310", "deg", "20", "deg", "3"],
                    ["zenith", "30", "deg", "40", "deg", "2"],
                    ["offset", "0", "deg", "10", "deg", "2"],
                ],
            }
        )


def test_build_axes_dict_from_cli_args_accepts_config_wrapped_axis_strings():
    axes = build_axes_dict_from_cli_args(
        {
            "axis": [
                ["azimuth 310 deg 20 deg 3 linear"],
                ["zenith 30 deg 40 deg 2"],
                ["offset 0 deg 10 deg 2"],
            ]
        }
    )

    assert axes["azimuth"]["range"] == [310.0, 20.0]
    assert axes[ZENITH_ANGLE_KEY]["binning"] == 2


def test_build_axes_dict_from_cli_args_rejects_multiple_direction_coordinate_systems():
    with pytest.raises(ValueError, match="Cannot define both azimuth/zenith and ha/dec axes"):
        build_axes_dict_from_cli_args(
            {
                "axis": [
                    ["azimuth", "310", "deg", "20", "deg", "3"],
                    ["zenith", "30", "deg", "40", "deg", "2"],
                    ["ha", "0", "deg", "360", "deg", "36"],
                    ["dec", "-90", "deg", "90", "deg", "18"],
                    ["offset", "0", "deg", "10", "deg", "2"],
                ]
            }
        )


def test_build_axes_dict_from_cli_args_rejects_invalid_density_unit():
    with pytest.raises(ValueError, match="direction_grid_density must be a float or quantity"):
        build_axes_dict_from_cli_args(
            {
                "direction_grid_density": "1.0 1/s",
                "axis": [
                    ["ha", "0", "deg", "360", "deg", "36"],
                    ["dec", "-90", "deg", "90", "deg", "18"],
                    ["offset", "0", "deg", "10", "deg", "2"],
                ],
            }
        )


def test_build_axes_dict_from_cli_args_rejects_invalid_density_type():
    with pytest.raises(TypeError, match="direction_grid_density must be a number"):
        build_axes_dict_from_cli_args(
            {
                "direction_grid_density": {"value": 1.0},
                "axis": [
                    ["ha", "0", "deg", "360", "deg", "36"],
                    ["dec", "-90", "deg", "90", "deg", "18"],
                    ["offset", "0", "deg", "10", "deg", "2"],
                ],
            }
        )


def test_build_job_grid_metadata_includes_job_context():
    metadata = build_job_grid_metadata(
        {
            "site": "North",
            "simulation_software": "corsika_sim_telarray",
            "direction_grid_density": 0.25,
            "axis": [
                ["ha", "0", "deg", "1", "deg", "2"],
                ["dec", "0", "deg", "1", "deg", "2"],
            ],
            "corsika_limits": "limits.ecsv",
        }
    )

    assert metadata["site"] == "North"
    assert metadata["simulation_software"] == "corsika_sim_telarray"
    assert metadata["coordinate_system"] == "ha_dec"
    assert metadata["direction_grid_density"] == pytest.approx(0.25)
    assert metadata["direction_grid_density_unit"] == "1/deg^2"
    assert "time_of_observation_utc" not in metadata
    assert metadata["corsika_limits"] == "limits.ecsv"


def test_build_job_grid_metadata_raises_for_hadec_without_site():
    with pytest.raises(ValueError, match="site is required"):
        build_job_grid_metadata(
            {
                "site": None,
                "simulation_software": "corsika_sim_telarray",
                "axis": [
                    ["ha", "0", "deg", "1", "deg", "2"],
                    ["dec", "0", "deg", "1", "deg", "2"],
                ],
                "corsika_limits": "limits.ecsv",
            }
        )


@patch("simtools.production_configuration.grid.serialize_job_grid")
@patch("simtools.production_configuration.grid.build_simulation_jobs", return_value=[])
def test_generate_job_grid_adds_default_metadata_to_ecsv_header(
    mock_build_simulation_jobs, mock_serialize_job_grid
):
    default_metadata = {"cta": {"activity": {"name": "production_generate_grid"}}}
    args = {
        "site": "North",
        "simulation_software": "corsika_sim_telarray",
        "axis": [
            ["azimuth", "0", "deg", "1", "deg", "2"],
            ["zenith", "0", "deg", "1", "deg", "2"],
        ],
    }

    generate_job_grid(args, "job_grid.ecsv", metadata=default_metadata)

    mock_build_simulation_jobs.assert_called_once_with(args, context=None)
    assert mock_serialize_job_grid.call_args.kwargs["metadata"] == {
        **default_metadata,
        "site": "North",
        "simulation_software": "corsika_sim_telarray",
        "coordinate_system": "horizontal",
        "direction_grid_density": None,
        "direction_grid_density_unit": None,
        "corsika_limits": None,
    }


@patch("simtools.production_configuration.grid.SiteModel")
def test_build_observing_location_uses_site_model(mock_site_model):
    model = mock_site_model.return_value
    model.get_parameter_value_with_unit.side_effect = [1 * u.deg, 2 * u.deg, 3 * u.m]

    location = build_observing_location("North", ["7.0.0"])

    assert isinstance(location, EarthLocation)
    mock_site_model.assert_called_once_with(model_version="7.0.0", site="North", context=None)


@patch("simtools.production_configuration.grid.SiteModel")
@patch("simtools.production_configuration.grid.build_observing_location")
@patch("simtools.production_configuration.grid.ProductionGridEngine")
def test_build_production_grid_engine_resolves_layout_name(
    mock_production_grid_engine,
    mock_build_observing_location,
    mock_site_model,
):
    location = EarthLocation(lat=1 * u.deg, lon=2 * u.deg, height=3 * u.m)
    mock_build_observing_location.return_value = location
    mock_site_model.return_value.get_nsb_integrated_flux.return_value = 0.42
    args_dict = {
        "site": "North",
        "array_layout_name": {"by_version": {"<7.0.0": "alpha", ">=7.0.0": "beta"}},
        "model_version": ["7.0.0"],
        "corsika_limits": "limits.ecsv",
        "axis": [
            ["azimuth", "310", "deg", "20", "deg", "3", "linear"],
            ["zenith", "30", "deg", "40", "deg", "2", "linear"],
            ["offset", "0", "deg", "10", "deg", "2", "linear"],
        ],
    }

    build_production_grid_engine(args_dict)

    mock_production_grid_engine.assert_called_once_with(
        axes={
            "offset": {"range": [0.0, 10.0], "binning": 2, "scaling": "linear", "units": "deg"},
            "azimuth": {
                "range": [310.0, 20.0],
                "binning": 3,
                "scaling": "linear",
                "units": "deg",
            },
            ZENITH_ANGLE_KEY: {
                "range": [30.0, 40.0],
                "binning": 2,
                "scaling": "linear",
                "units": "deg",
            },
        },
        coordinate_system="horizontal",
        observing_location=location,
        lookup_table="limits.ecsv",
        array_layout_name="beta",
        lookup_nsb_rate=0.42,
    )
    mock_build_observing_location.assert_called_once_with(
        site="North", model_version="7.0.0", context=None
    )
    mock_site_model.assert_called_once_with(model_version="7.0.0", site="North", context=None)


def test_build_production_grid_engine_raises_without_site_for_nsb_rate():
    with pytest.raises(ValueError, match="site and model_version are required"):
        build_production_grid_engine(
            {
                "model_version": ["7.0.0"],
                "corsika_limits": None,
                "axis": [
                    ["azimuth", "310", "deg", "20", "deg", "3", "linear"],
                    ["zenith", "30", "deg", "40", "deg", "2", "linear"],
                    ["offset", "0", "deg", "10", "deg", "2", "linear"],
                ],
            }
        )


@pytest.mark.parametrize(
    ("energy_range", "expected_length"),
    [
        ((30 * u.GeV, 100 * u.GeV), 1),
        ([[30 * u.GeV, 100 * u.GeV], [100 * u.GeV, 1 * u.TeV]], 2),
    ],
)
def test_normalize_energy_ranges_accepts_supported_shapes(energy_range, expected_length):
    assert len(normalize_energy_ranges(energy_range)) == expected_length


def test_calculate_log_energy_midpoint_returns_geometric_mean():
    midpoint = calculate_log_energy_midpoint((10 * u.GeV, 1 * u.TeV))

    assert_quantity_allclose(midpoint, 100 * u.GeV)


def test_calculate_scaled_showers_per_run_returns_baseline_without_power_law():
    assert calculate_scaled_showers_per_run((30 * u.GeV, 100 * u.GeV), 5) == 5


def test_calculate_scaled_showers_per_run_raises_for_invalid_power_law_tuple():
    with pytest.raises(ValueError, match="exactly two values"):
        calculate_scaled_showers_per_run(
            (30 * u.GeV, 100 * u.GeV), 5, showers_per_run_power_law=(1.0,)
        )


def test_calculate_scaled_showers_per_run_scales_from_midpoint_energy():
    scaled_showers_per_run = calculate_scaled_showers_per_run(
        (10 * u.GeV, 1 * u.TeV),
        5,
        showers_per_run_power_law=(1.0, 10 * u.GeV),
    )

    assert scaled_showers_per_run == 50


@patch("simtools.production_configuration.grid.np.ceil", return_value=0)
def test_calculate_scaled_showers_per_run_raises_when_scaled_value_is_below_one(mock_ceil):
    with pytest.raises(ValueError, match="at least 1"):
        calculate_scaled_showers_per_run(
            (10 * u.GeV, 1 * u.TeV),
            5,
            showers_per_run_power_law=(1.0, 10 * u.GeV),
        )
    mock_ceil.assert_called_once()


def test_calculate_zenith_scaled_showers_per_run_returns_baseline_for_fixed_mode():
    assert calculate_zenith_scaled_showers_per_run(20 * u.deg, 1000, "fixed") == 1000


def test_calculate_zenith_scaled_showers_per_run_scales_with_cosine():
    expected = int(np.ceil(1000 * np.round(np.cos(np.radians(60)), decimals=12)))
    assert (
        calculate_zenith_scaled_showers_per_run(60 * u.deg, 1000, COSINE_ZENITH_SCALING) == expected
    )


def test_calculate_zenith_scaled_showers_per_run_raises_at_zenith_90():
    with pytest.raises(ValueError, match="at least 1"):
        calculate_zenith_scaled_showers_per_run(90 * u.deg, 1000, COSINE_ZENITH_SCALING)


def test_calculate_zenith_scaled_showers_per_run_raises_for_invalid_mode():
    with pytest.raises(ValueError, match="Unknown showers_per_run_scaling mode"):
        calculate_zenith_scaled_showers_per_run(20 * u.deg, 1000, "invalid_mode")


def test_scale_energy_max_for_zenith_angle_returns_original_without_scaling_index():
    energy_range = (30 * u.GeV, 100 * u.GeV)

    assert scale_energy_max_for_zenith_angle(60 * u.deg, energy_range, None) == energy_range


def test_scale_energy_max_for_zenith_angle_scales_max_energy():
    scaled = scale_energy_max_for_zenith_angle(
        60 * u.deg,
        (30 * u.GeV, 100 * u.GeV),
        (-2.0, 100 * u.GeV),
    )

    assert_quantity_allclose(scaled[0], 30 * u.GeV)
    assert_quantity_allclose(scaled[1], 400 * u.GeV)


def test_scale_energy_max_for_zenith_angle_returns_none_if_scaled_max_below_min():
    assert (
        scale_energy_max_for_zenith_angle(
            60 * u.deg,
            (30 * u.GeV, 100 * u.GeV),
            (2.0, 100 * u.GeV),
        )
        is None
    )


def test_scale_energy_max_for_zenith_angle_raises_for_negative_index_at_zenith_90():
    with pytest.raises(ValueError, match="energy_max_scaling"):
        scale_energy_max_for_zenith_angle(90 * u.deg, (30 * u.GeV, 100 * u.GeV), (-2.0, None))


@pytest.mark.parametrize(
    ("power_law", "expected_index", "expected_energy"),
    [
        (["1.0", "100", "GeV"], 1.0, 100 * u.GeV),
        ("1.0 100 GeV", 1.0, 100 * u.GeV),
        (["-1.0 100 TeV"], -1.0, 100 * u.TeV),
    ],
)
def test_resolve_shower_params_converts_showers_per_run_power_law(
    power_law, expected_index, expected_energy
):
    (
        showers_per_run,
        power_law_parsed,
        showers_per_run_scaling,
        total_showers,
        total_showers_scaling,
    ) = _resolve_shower_params({"showers_per_run": 5, "showers_per_run_power_law": power_law})

    assert showers_per_run == 5
    assert power_law_parsed[0] == pytest.approx(expected_index)
    assert power_law_parsed[1] == expected_energy
    assert showers_per_run_scaling == "fixed"
    assert total_showers is None
    assert total_showers_scaling == "fixed"


def test_build_rows_for_point_skips_energy_ranges_below_threshold():
    rows = _build_rows_for_point(
        point_base={"primary": "gamma", ZENITH_ANGLE_KEY: 20 * u.deg},
        energy_ranges=[(30 * u.GeV, 40 * u.GeV), (50 * u.GeV, 100 * u.GeV)],
        lower_energy_threshold=45 * u.GeV,
        showers_per_run=5,
        showers_per_run_power_law=None,
        number_of_runs=2,
        run_number=10,
        run_allocation=_RunAllocationConfig(),
    )

    assert [row["run_number"] for row in rows] == [10, 11]
    assert all(row["energy_min"] == 50 * u.GeV for row in rows)
    assert all(row["showers_per_run"] == 5 for row in rows)


def test_build_rows_for_point_rounds_total_showers_up_with_warning(caplog):
    caplog.set_level("WARNING")

    rows = _build_rows_for_point(
        point_base={"primary": "gamma", ZENITH_ANGLE_KEY: 20 * u.deg},
        energy_ranges=[(30 * u.GeV, 100 * u.GeV)],
        lower_energy_threshold=None,
        showers_per_run=1000,
        showers_per_run_power_law=None,
        number_of_runs=2,
        run_number=1,
        run_allocation=_RunAllocationConfig(total_showers=2500),
    )

    assert [row["showers_per_run"] for row in rows] == [1000, 1000, 1000]
    assert [row["run_number"] for row in rows] == [1, 2, 3]
    assert "adjusting to 3000 to keep equal showers per run" in caplog.text


def test_build_rows_for_point_preserves_zenith_scaling_with_showers_cap():
    rows = _build_rows_for_point(
        point_base={ZENITH_ANGLE_KEY: 60 * u.deg},
        energy_ranges=[(30 * u.GeV, 100 * u.GeV)],
        lower_energy_threshold=None,
        showers_per_run=1000,
        showers_per_run_power_law=(1.0, 100 * u.GeV),
        showers_per_run_scaling=COSINE_ZENITH_SCALING,
        number_of_runs=1,
        run_number=1,
        run_allocation=_RunAllocationConfig(
            total_showers=2500,
            max_showers_per_run=1000,
        ),
    )

    assert [row["showers_per_run"] for row in rows] == [250] * 10


def test_build_rows_for_point_caps_total_showers_rounding_warnings(caplog):
    caplog.set_level("WARNING")

    _build_rows_for_point(
        point_base={"primary": "gamma", ZENITH_ANGLE_KEY: 20 * u.deg},
        energy_ranges=[(30 * u.GeV, 100 * u.GeV)] * 4,
        lower_energy_threshold=None,
        showers_per_run=1000,
        showers_per_run_power_law=None,
        number_of_runs=1,
        run_number=1,
        run_allocation=_RunAllocationConfig(
            total_showers=2500,
            rounding_warning_state={
                "emitted_warnings": 0,
                "max_warnings": 2,
                "suppression_warning_emitted": False,
            },
        ),
    )

    detailed_warning_messages = [
        record.message
        for record in caplog.records
        if "adjusting to 3000 to keep equal showers per run" in record.message
    ]
    suppression_warning_messages = [
        record.message
        for record in caplog.records
        if "further total_showers rounding warnings will be suppressed" in record.message
    ]

    assert len(detailed_warning_messages) == 2
    assert len(suppression_warning_messages) == 1


def test_build_rows_for_point_scales_total_showers_with_zenith_scaled():
    rows = _build_rows_for_point(
        point_base={ZENITH_ANGLE_KEY: 60 * u.deg},
        energy_ranges=[(30 * u.GeV, 100 * u.GeV)],
        lower_energy_threshold=None,
        showers_per_run=200,
        showers_per_run_power_law=None,
        number_of_runs=1,
        run_number=1,
        run_allocation=_RunAllocationConfig(
            total_showers=2500,
            total_showers_scaling="zenith_scaled",
        ),
    )

    assert [row["showers_per_run"] for row in rows] == [200, 200]
    assert [row["run_number"] for row in rows] == [1, 2]


def test_generate_observation_points_from_axes_adds_lookup_limits():
    corsika_limits = Mock()
    corsika_limits.interpolate_point.return_value = {
        "lower_energy_limit": 0.05,
        "upper_radius_limit": 150.0,
        "viewcone_radius": 3.0,
    }
    corsika_limits.lookup_field_units = {
        "lower_energy_limit": u.TeV,
        "upper_radius_limit": u.m,
        "viewcone_radius": u.deg,
    }

    points = _generate_observation_points_from_axes(
        [180 * u.deg],
        [20 * u.deg],
        corsika_limits,
        nsb_rate=0.37,
    )

    assert len(points) == 1
    corsika_limits.interpolate_point.assert_called_once_with(
        20 * u.deg,
        180 * u.deg,
        nsb=0.37,
    )
    assert points[0]["nsb_rate"] == pytest.approx(0.37)
    assert_quantity_allclose(points[0]["lower_energy_limit"], 0.05 * u.TeV)
    assert_quantity_allclose(points[0]["upper_radius_limit"], 150 * u.m)
    assert_quantity_allclose(points[0]["viewcone_radius"], 3 * u.deg)


@patch("simtools.production_configuration.grid.SiteModel")
@patch("simtools.production_configuration.grid.build_production_grid_engine")
def test_generate_observation_grids_per_layout_uses_shared_axes_and_skips_duplicate_layouts(
    mock_build_production_grid_engine,
    mock_site_model,
):
    mock_build_production_grid_engine.return_value.generate_simulation_grid.return_value = [
        {"azimuth": 0 * u.deg}
    ]
    mock_site_model.return_value.get_nsb_integrated_flux.return_value = 0.31

    observation_grids, resolved_layout_names = _generate_observation_grids_per_layout(
        _horizontal_shared_axis_args({"<7.0.0": "alpha", ">=7.0.0": "alpha"}),
        _two_model_versions_grid_axes(),
    )

    assert observation_grids == {
        "6.3.0": [{"azimuth": 0 * u.deg}],
        "7.0.0": [{"azimuth": 0 * u.deg}],
    }
    assert resolved_layout_names == {"6.3.0": "alpha", "7.0.0": "alpha"}
    mock_build_production_grid_engine.assert_called_once()
    assert mock_site_model.call_count == 2


@patch("simtools.production_configuration.grid._generate_observation_grids_per_layout")
def test_build_simulation_jobs_expands_runs_from_observation_grid(
    mock_generate_observation_grids_per_layout,
    run_context,
):
    mock_generate_observation_grids_per_layout.return_value = _observation_grid_return(
        {
            "azimuth": 180 * u.deg,
            ZENITH_ANGLE_KEY: 20 * u.deg,
            "ha": 123 * u.deg,
            "dec": -45 * u.deg,
            "lower_energy_limit": 40 * u.GeV,
            "upper_radius_limit": 100 * u.m,
            "viewcone_radius": 2 * u.deg,
        }
    )
    args_dict = _base_simulation_jobs_args()
    args_dict[NUMBER_OF_RUNS_ARGUMENT] = 2
    rows = build_simulation_jobs(args_dict, context=run_context)

    assert [row["run_number"] for row in rows] == [11, 12]
    assert rows[0]["array_layout_name"] == "alpha"
    assert rows[0]["core_scatter_max"] == 100 * u.m
    assert rows[0]["view_cone_min"] == 0 * u.deg
    assert rows[0]["view_cone_max"] == 2 * u.deg
    assert rows[0]["showers_per_run"] == 5
    assert rows[0]["ha"] == 123 * u.deg
    assert rows[0]["dec"] == -45 * u.deg


@patch("simtools.production_configuration.grid._generate_observation_grids_per_layout")
def test_build_simulation_jobs_from_horizontal_grid_does_not_add_hadec(
    mock_generate_observation_grids_per_layout,
    run_context,
):
    mock_generate_observation_grids_per_layout.return_value = _observation_grid_return(
        {
            "azimuth": 180 * u.deg,
            ZENITH_ANGLE_KEY: 20 * u.deg,
            "lower_energy_limit": 40 * u.GeV,
            "upper_radius_limit": 100 * u.m,
            "viewcone_radius": 2 * u.deg,
        }
    )
    rows = build_simulation_jobs(_base_simulation_jobs_args(), context=run_context)

    assert "ha" not in rows[0]
    assert "dec" not in rows[0]
    assert "corsika_hadronic_transition_energy" not in rows[0]


@patch("simtools.production_configuration.grid._generate_observation_grids_per_layout")
def test_build_simulation_jobs_propagates_explicit_hadronic_transition_energy(
    mock_generate_observation_grids_per_layout,
    run_context,
):
    mock_generate_observation_grids_per_layout.return_value = _observation_grid_return(
        {
            "azimuth": 180 * u.deg,
            ZENITH_ANGLE_KEY: 20 * u.deg,
            "lower_energy_limit": 40 * u.GeV,
            "upper_radius_limit": 100 * u.m,
            "viewcone_radius": 2 * u.deg,
        }
    )
    args_dict = _base_simulation_jobs_args()
    args_dict["corsika_hadronic_transition_energy"] = 120 * u.GeV

    rows = build_simulation_jobs(args_dict, context=run_context)

    assert rows[0]["corsika_hadronic_transition_energy"] == 120 * u.GeV


@patch("simtools.production_configuration.grid._generate_observation_grids_per_layout")
def test_build_simulation_jobs_raises_for_total_showers_and_number_of_runs(
    mock_generate_observation_grids_per_layout,
):
    mock_generate_observation_grids_per_layout.return_value = ({"alpha": []}, {"6.3.0": "alpha"})
    args_dict = _base_simulation_jobs_args()
    args_dict[NUMBER_OF_RUNS_ARGUMENT] = 2
    args_dict[TOTAL_SHOWERS_ARGUMENT] = 100

    with pytest.raises(ValueError, match="total_showers and number_of_runs"):
        build_simulation_jobs(args_dict)


@pytest.mark.parametrize("maximum", [0, -1])
def test_build_simulation_jobs_rejects_non_positive_maximum(maximum):
    args_dict = _base_simulation_jobs_args()
    args_dict[TOTAL_SHOWERS_ARGUMENT] = 2500
    args_dict[NUMBER_OF_RUNS_ARGUMENT] = None
    args_dict[MAX_SHOWERS_PER_RUN_ARGUMENT] = maximum

    with pytest.raises(
        ValueError,
        match=f"{MAX_SHOWERS_PER_RUN_ARGUMENT} must be a positive integer",
    ):
        build_simulation_jobs(args_dict)


@pytest.mark.parametrize(
    ("axis_spec", "match"),
    [
        ("zenith 2", "at least an axis name"),
        ("badaxis 0 deg 90 deg 3", "Unknown axis"),
        ("zenith 3 linear", "missing its range"),  # name + binning + scaling only
        ("zenith 0 deg 90 deg foo", "binning must be an integer"),
    ],
)
def test_parse_axis_spec_raises_for_invalid_definitions(axis_spec, match):
    with pytest.raises(ValueError, match=match):
        _parse_axis_spec(axis_spec)


def test_build_axes_dict_from_cli_args_raises_without_direction_axes():
    with pytest.raises(ValueError, match="azimuth/zenith or both ha/dec"):
        build_axes_dict_from_cli_args(
            {
                "axis": [
                    ["offset", "0", "deg", "10", "deg", "2"],
                ]
            }
        )


def test_build_axes_dict_from_cli_args_raises_for_missing_required_axes():
    with pytest.raises(ValueError, match="Missing required shared axis"):
        build_axes_dict_from_cli_args(
            {
                "axis": [
                    ["azimuth", "310", "deg", "20", "deg", "3"],
                    ["zenith", "30", "deg", "40", "deg", "2"],
                ]
            }
        )


def test_build_rows_for_point_skips_when_effective_total_showers_is_zero():
    rows = _build_rows_for_point(
        point_base={"primary": "gamma", ZENITH_ANGLE_KEY: 20 * u.deg},
        energy_ranges=[(30 * u.GeV, 100 * u.GeV)],
        lower_energy_threshold=None,
        showers_per_run=1000,
        showers_per_run_power_law=None,
        number_of_runs=1,
        run_number=1,
        run_allocation=_RunAllocationConfig(total_showers=0),
    )

    assert rows == []


def test_renumber_job_rows_applies_run_number_offset():
    job_rows = [{"run_number": 10}, {"run_number": 10}, {"run_number": 11}]

    assert [row["run_number"] for row in renumber_job_rows(job_rows, run_number_offset=42)] == [
        43,
        44,
        45,
    ]


def test_build_job_grid_summary_includes_used_configured_and_lookup_values():
    rows = [
        {
            "energy_min": 40 * u.GeV,
            "energy_max": 100 * u.GeV,
            "configured_energy_min": 30 * u.GeV,
            "configured_energy_max": 100 * u.GeV,
            "energy_min_lookup_limit": 40 * u.GeV,
            "core_scatter_max": 150 * u.m,
            "configured_core_scatter_max": 200 * u.m,
            "lookup_core_scatter_max": 150 * u.m,
            "view_cone_min": 0 * u.deg,
            "view_cone_max": 2 * u.deg,
            "configured_view_cone_min": 0 * u.deg,
            "configured_view_cone_max": 5 * u.deg,
            "lookup_view_cone_max": 2 * u.deg,
            "showers_per_run": 500,
            "configured_showers_per_run": 1000,
        },
        {
            "energy_min": 50 * u.GeV,
            "energy_max": 100 * u.GeV,
            "configured_energy_min": 30 * u.GeV,
            "configured_energy_max": 100 * u.GeV,
            "energy_min_lookup_limit": 50 * u.GeV,
            "core_scatter_max": 200 * u.m,
            "configured_core_scatter_max": 200 * u.m,
            "lookup_core_scatter_max": 250 * u.m,
            "view_cone_min": 1 * u.deg,
            "view_cone_max": 5 * u.deg,
            "configured_view_cone_min": 1 * u.deg,
            "configured_view_cone_max": 5 * u.deg,
            "lookup_view_cone_max": 10 * u.deg,
            "showers_per_run": 1000,
            "configured_showers_per_run": 1000,
        },
    ]

    summary = build_job_grid_summary(rows)

    assert summary["simulation_rows"] == 2
    assert summary["energy_min_used"] == "[40, 50] GeV"
    assert summary["core_scatter_max_used"] == "[150, 200] m"
    assert summary["core_scatter_max_configured"] == "200 m"
    assert summary["view_cone_max_lookup_limit"] == "[2, 10] deg"
    assert summary["showers_per_run_min"] == 500
    assert summary["showers_per_run_max"] == 1000
    assert summary["showers_per_run_used"] == "[500, 1000]"
    assert summary["showers_per_run_configured_max"] == 1000
    assert summary["showers_per_run_configured"] == "1000"
    assert summary["total_showers"] == 1500


def test_generate_simulation_grid_keeps_horizontal_coordinates_for_hadec_axes():
    engine = _make_engine(
        axes={
            "ha": {"range": [0, 0], "binning": 1, "scaling": "linear", "units": "deg"},
            "dec": {"range": [0, 0], "binning": 1, "scaling": "linear", "units": "deg"},
        },
        coordinate_system="ha_dec",
        observing_location=DEFAULT_OBSERVING_LOCATION,
        lookup_table=None,
    )

    simulation_grid = engine.generate_simulation_grid()

    assert "zenith_angle" in simulation_grid[0]
    assert "azimuth" in simulation_grid[0]
    assert "ha" in simulation_grid[0]
    assert "dec" in simulation_grid[0]


def test_generate_simulation_grid_keeps_horizontal_grid_horizontal():
    engine = _make_engine(
        axes={
            "azimuth": {"range": [180, 180], "binning": 1, "scaling": "linear", "units": "deg"},
            "zenith_angle": {
                "range": [20, 20],
                "binning": 1,
                "scaling": "linear",
                "units": "deg",
            },
        },
        coordinate_system="horizontal",
        observing_location=DEFAULT_OBSERVING_LOCATION,
        lookup_table=None,
    )

    simulation_grid = engine.generate_simulation_grid()

    assert "zenith_angle" in simulation_grid[0]
    assert "azimuth" in simulation_grid[0]
    assert "ha" not in simulation_grid[0]
    assert "dec" not in simulation_grid[0]


def test_generate_target_values_supports_log_and_inverse_cos_scaling():
    engine = ProductionGridEngine(
        axes={
            "axes": {
                "energy": {"range": [1, 100], "binning": 3, "scaling": "log", "units": "TeV"},
                "zenith_angle": {
                    "range": [0, 60],
                    "binning": 3,
                    "scaling": "1/cos",
                    "units": "deg",
                },
            }
        }
    )

    assert_quantity_allclose(engine.target_values["energy"], [1, 10, 100] * u.TeV)
    assert len(engine.target_values["zenith_angle"]) == 3


def test_add_lookup_limits_to_point_uses_engine_lookup_nsb_rate():
    engine = _make_engine(lookup_table=None, lookup_nsb_rate=1.7)
    engine.lookup_table = "limits.ecsv"
    engine._interpolate_limits_for_point = Mock(
        return_value={
            "lower_energy_threshold": 0.02,
            "upper_scatter_radius": 120.0,
            "viewcone_radius": 3.5,
        }
    )
    point = {}

    engine._add_lookup_limits_to_point(point, zenith=20.0, azimuth=180.0)

    engine._interpolate_limits_for_point.assert_called_once_with(
        zenith=20.0, azimuth=180.0, nsb=1.7
    )
    assert point["nsb_rate"] == pytest.approx(1.7)
    assert_quantity_allclose(point["lower_energy_limit"], 0.02 * u.TeV)
    assert_quantity_allclose(point["upper_radius_limit"], 120 * u.m)
    assert_quantity_allclose(point["viewcone_radius"], 3.5 * u.deg)


def test_add_lookup_limits_to_point_raises_without_nsb_rate():
    engine = _make_engine(lookup_table=None, lookup_nsb_rate=None)
    engine.lookup_table = "limits.ecsv"
    engine._interpolate_limits_for_point = Mock(return_value={"lower_energy_limit": 0.02})

    with pytest.raises(ValueError, match="nsb_rate is required"):
        engine._add_lookup_limits_to_point({}, zenith=20.0, azimuth=180.0)


def test_generate_extra_axis_combinations_returns_mesh_for_remaining_axes():
    engine = ProductionGridEngine(
        axes={
            "axes": {
                "zenith_angle": {"range": [20, 20], "binning": 1, "units": "deg"},
                "nsb_level": {"range": [1, 2], "binning": 2, "units": "1"},
                "offset": {"range": [0, 1], "binning": 2, "units": "deg"},
            }
        }
    )

    keys, units, combinations = engine._generate_extra_axis_combinations(("zenith_angle",))

    assert keys == ["nsb_level", "offset"]
    assert units == [u.dimensionless_unscaled, u.deg]
    assert combinations.shape == (4, 2)


def test_create_circular_binning_uses_directed_span_from_start_to_end():
    engine = _make_engine()

    binning = engine.create_circular_binning((10, 350), 3)

    assert np.allclose(binning, [10, 180, 350])


def test_generate_horizontal_grid_uses_adaptive_azimuth_bins_per_zenith_row():
    engine = ProductionGridEngine(
        axes={
            "axes": {
                "zenith_angle": {"range": [0, 60], "binning": 61, "units": "deg"},
                "azimuth": {
                    "range": [0, 240],
                    "binning": 10,
                    "units": "deg",
                    "direction_grid_density": 0.05,
                },
                "nsb_level": {"range": [1, 1], "binning": 1, "units": "1"},
            }
        },
        coordinate_system="horizontal",
    )

    grid = engine._generate_horizontal_grid()

    assert len(grid) > 0

    azimuth_counts_by_zenith = {}
    for point in grid:
        zenith_value = round(point["zenith_angle"].to_value(u.deg), 6)
        azimuth_counts_by_zenith.setdefault(zenith_value, set()).add(
            round(point["azimuth"].to_value(u.deg), 6)
        )

    assert len(azimuth_counts_by_zenith[0.0]) == 1
    assert len(azimuth_counts_by_zenith[30.0]) > len(azimuth_counts_by_zenith[0.0])
    assert len(azimuth_counts_by_zenith[60.0]) > len(azimuth_counts_by_zenith[30.0])


def test_generate_hadec_grid_uses_adaptive_ha_bins_per_dec_strip():
    engine = ProductionGridEngine(
        axes={
            "ha": {"range": [0, 360], "binning": 36, "units": "deg", "direction_grid_density": 1.0},
            "dec": {"range": [-60, 60], "binning": 13, "units": "deg"},
            "offset": {"range": [0, 0], "binning": 1, "units": "deg"},
        },
        coordinate_system="ha_dec",
    )

    assert engine._is_adaptive_hadec_density_enabled()
    grid = engine._generate_adaptive_hadec_grid(include_horizontal_coordinates=False)

    assert len(grid) > 0

    ha_counts_by_dec = {}
    for point in grid:
        dec_val = round(point["dec"].to_value(u.deg), 6)
        ha_counts_by_dec.setdefault(dec_val, 0)
        ha_counts_by_dec[dec_val] += 1

    # Near equator (dec=0) should have more HA points than near the pole (dec=60 deg)
    assert ha_counts_by_dec[0.0] > ha_counts_by_dec[60.0]


def test_generate_hadec_grid_adaptive_density_keeps_only_visible_nodes():
    engine = ProductionGridEngine(
        axes=_adaptive_hadec_axes(),
        coordinate_system="ha_dec",
        observing_location=DEFAULT_OBSERVING_LOCATION,
    )

    grid = engine._generate_adaptive_hadec_grid(include_horizontal_coordinates=True)

    assert len(grid) > 0
    assert all(point["zenith_angle"].to_value(u.deg) <= 90.0 for point in grid)


def test_generate_hadec_grid_adaptive_density_applies_zenith_constraint():
    engine = ProductionGridEngine(
        axes=_adaptive_hadec_axes(local_zenith_range=[0, 70]),
        coordinate_system="ha_dec",
        observing_location=DEFAULT_OBSERVING_LOCATION,
    )

    grid = engine._generate_adaptive_hadec_grid(include_horizontal_coordinates=True)

    assert len(grid) > 0
    assert all(point["zenith_angle"].to_value(u.deg) <= 70.0 for point in grid)


def test_generate_hadec_grid_adaptive_density_applies_azimuth_constraint():
    engine = ProductionGridEngine(
        axes=_adaptive_hadec_axes(
            local_zenith_range=[0, 70],
            local_azimuth_range=[300, 60],
        ),
        coordinate_system="ha_dec",
        observing_location=DEFAULT_OBSERVING_LOCATION,
    )

    grid = engine._generate_adaptive_hadec_grid(include_horizontal_coordinates=True)

    assert len(grid) > 0
    assert all(
        (point["azimuth"].to_value(u.deg) >= 300.0) or (point["azimuth"].to_value(u.deg) <= 60.0)
        for point in grid
    )


def test_create_circular_binning_treats_full_circle_range_as_full_span():
    engine = _make_engine()

    binning = engine.create_circular_binning((0, 360), 4)

    assert np.allclose(binning, [0, 90, 180, 270])


def test_convert_altaz_to_hadec_returns_local_coordinates():
    engine = _make_engine(
        observing_location=DEFAULT_OBSERVING_LOCATION,
    )

    hadec = engine.convert_altaz_to_hadec(45 * u.deg, 180 * u.deg)

    assert_quantity_allclose(hadec.ha.to(u.deg), 0 * u.deg, atol=1e-10 * u.deg)
    assert_quantity_allclose(hadec.dec.to(u.deg), -16.24 * u.deg, atol=1e-10 * u.deg)


def test_convert_coordinates_drops_horizontal_coordinates_when_requested():
    engine = _make_engine(coordinate_system="ha_dec")
    engine.convert_altaz_to_hadec = Mock(return_value=Mock(ha=10 * u.deg, dec=-20 * u.deg))
    points = [{"zenith_angle": 20 * u.deg, "azimuth": 180 * u.deg}]

    converted = engine.convert_coordinates(points, keep_horizontal_coordinates=False)

    assert "zenith_angle" not in converted[0]
    assert "azimuth" not in converted[0]
    assert_quantity_allclose(converted[0]["ha"], 10 * u.deg)
    assert_quantity_allclose(converted[0]["dec"], -20 * u.deg)


def test_convert_coordinates_keeps_points_without_horizontal_values():
    engine = _make_engine(coordinate_system="ha_dec")
    points = [{"ha": 10 * u.deg, "dec": -20 * u.deg}]

    converted = engine.convert_coordinates(points, keep_horizontal_coordinates=False)

    assert converted == points


@patch.object(ProductionGridEngine, "_prepare_lookup_table_limits_for_point_interpolation")
@patch("simtools.production_configuration.grid.CorsikaLimitsLookup")
def test_init_with_lookup_prepares_point_interpolation(
    mock_corsika_limits_lookup,
    mock_prepare_lookup_table_limits,
):
    ProductionGridEngine(
        axes={"axes": {"ha": {"range": [0, 0], "binning": 1, "units": "deg"}}},
        coordinate_system="ha_dec",
        lookup_table="limits.ecsv",
        array_layout_name="alpha",
    )

    mock_corsika_limits_lookup.assert_called_once_with(
        lookup_table="limits.ecsv",
        array_layout_name="alpha",
    )
    mock_prepare_lookup_table_limits.assert_called_once_with()


def test_directed_circular_span_degrees_returns_directed_span_for_non_wrapping_range():
    assert directed_circular_span_degrees((10, 110)) == pytest.approx(100.0)


def test_directed_circular_span_degrees_treats_full_circle_as_360_degrees():
    assert directed_circular_span_degrees((0, 360)) == pytest.approx(360.0)
    assert directed_circular_span_degrees((15, 375)) == pytest.approx(360.0)


def test_ceil_with_tolerance_rounds_near_integer_without_overshooting():
    assert ceil_with_tolerance(10.00000000001) == 10


def test_ceil_with_tolerance_uses_ceiling_for_non_integer_values():
    assert ceil_with_tolerance(10.2) == 11

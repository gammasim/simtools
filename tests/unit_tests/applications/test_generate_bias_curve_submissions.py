from astropy import units as u

import simtools.applications.generate_bias_curve_submissions as app
from simtools.configuration.commandline_parser import CommandLineParser


def _required_arguments():
    return [
        "--site",
        "North",
        "--model_version",
        "7.0.0",
        "--array_layout_name",
        "LSTN-01",
        "--azimuth_angle",
        "0",
        "--zenith_angle",
        "20",
        "--showers_per_run",
        "10000",
        "--core_scatter",
        "20 1900 m",
        "--view_cone",
        "0 deg 5 deg",
        "--number_of_runs",
        "10",
    ]


def test_add_arguments_uses_bias_curve_defaults():
    parser = CommandLineParser()
    parser.add_argument_definitions(app._ARGUMENTS)

    args = parser.parse_args(_required_arguments())

    assert args.nsb_energy_range == (20 * u.MeV, 25 * u.MeV)
    assert args.proton_energy_range == (2 * u.GeV, 2000 * u.GeV)
    assert args.nsb_scaling_factor == 2
    assert args.trigger_thresholds is None
    assert args.core_scatter == (20, 1900 * u.m)
    assert args.view_cone == (0 * u.deg, 5 * u.deg)
    assert args.array_layout_name == ["LSTN-01"]
    assert not hasattr(args, "telescope")

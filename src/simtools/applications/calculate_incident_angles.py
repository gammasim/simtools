#!/usr/bin/env python
r"""
Calculate incident angles using sim_telarray ray tracing.

Creates photon files with additional columns for incident angles calculation.
This application runs a version of sim_telarray compiled with the preprocessing
flag -DDEBUG_TRACE 99. Outputs files and histograms of the incidence angles at
the focal plane, primary mirror, and if available, secondary mirror.

Example usage
-------------

.. code-block:: console

    simtools-calculate-incident-angles \
        --zenith 20 \
        --off_axis_angles 0 1 2 3 4 \
        --source_distance 10 \
        --number_of_rays 10000 \
        --model_version 6.0.0 \
        --telescope MSTN-04 \
        --site North

Command line arguments
----------------------

zenith (float, optional)
    Zenith angle in degrees (default: 20.0).
off_axis_angles (float, optional)
    One or more off-axis angles in degrees (space-separated).
source_distance (float, optional)
    Source distance in kilometers (default: 10.0).
number_of_rays (int, optional)
    Number of star photons to trace per run (default: 10000).
perfect_mirror (flag, optional)
    Assume perfect mirror shape/alignment/reflection.
overwrite_rdna (flag, optional)
    Overwrite mirror_reflection_random_angle with 0 deg.
mirror_reflection_random_angle (float, optional)
    Set mirror_reflection_random_angle in degrees (overrides overwrite_rdna).
algn (float, optional)
    Accuracy parameter for mirror alignment distributions.

The application writes:
- imaging list (photons) file
- stars list file
- a histogram of incident angles (PNG)
- a results table in ECSV format
"""

import logging
from pathlib import Path

import astropy.units as u

from simtools.configuration import configurator
from simtools.ray_tracing.incident_angles import IncidentAnglesCalculator
from simtools.visualization.plot_incident_angles import plot_incident_angles


def _parse(label):
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=label,
        description="Calculate incident angles using sim_telarray ray tracing.",
    )

    config.parser.add_argument(
        "--zenith",
        help="Zenith angle in degrees",
        type=float,
        default=20.0,
        required=False,
    )
    config.parser.add_argument(
        "--off_axis_angles",
        help="One or more off-axis angles in degrees (space-separated)",
        type=float,
        nargs="+",
        required=False,
    )
    config.parser.add_argument(
        "--source_distance",
        help="Source distance in kilometers",
        type=float,
        default=10.0,
        required=False,
    )
    config.parser.add_argument(
        "--number_of_rays",
        help="Number of star photons to trace (per run)",
        type=int,
        default=10000,
        required=False,
    )
    config.parser.add_argument(
        "--camera_shift",
        help="Camera shift along optical axis",
        type=float,
        default=0.0,
        required=False,
    )
    config.parser.add_argument(
        "--overwrite_rdna",
        help="Overwrite random reflection angle",
        action="store_true",
        required=False,
    )
    config.parser.add_argument(
        "--mirror_reflection_random_angle",
        help="Set mirror_reflection_random_angle in degrees (overrides overwrite_rdna)",
        type=float,
        required=False,
    )
    config.parser.add_argument(
        "--algn",
        help="Accuracy of mirror alignment",
        type=float,
        default=0.005,
        required=False,
    )
    config.parser.add_argument(
        "--use_prod4",
        help="Use prod-4 config files (SST: prod-5) instead of prod-6",
        action="store_true",
        required=False,
    )
    config.parser.add_argument(
        "--flip_mirror_layout",
        help="Flip mirror layout x/y",
        action="store_true",
        required=False,
    )
    config.parser.add_argument(
        "--no_optimisation",
        help="No optimisation in rx. Containment radii around c.o.g.",
        action="store_true",
        required=False,
    )
    config.parser.add_argument(
        "--perfect_mirror",
        help="Assume perfect mirror shape/alignment/reflection",
        action="store_true",
        required=False,
    )
    calc_group = config.parser.add_mutually_exclusive_group(required=False)
    calc_group.add_argument(
        "--calculate_primary_secondary_angles",
        dest="calculate_primary_secondary_angles",
        action="store_true",
        help="Also compute angles of incidence on primary and secondary mirrors",
    )
    calc_group.add_argument(
        "--no-calculate_primary_secondary_angles",
        dest="calculate_primary_secondary_angles",
        action="store_false",
        help="Do not compute angles of incidence on primary and secondary mirrors",
    )
    config.parser.set_defaults(calculate_primary_secondary_angles=True)
    return config.initialize(
        db_config=True,
        simulation_model=["telescope", "site", "model_version"],
    )


def main():
    """Application to calculate incident angles using ray tracing."""
    label = Path(__file__).stem
    args_dict, db_config = _parse(label)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info("Starting calculation of incident angles")

    output_base = Path(args_dict.get("output_path", "./"))
    output_dir = (
        output_base / label if not args_dict.get("use_plain_output_path", False) else output_base
    )
    base_label = args_dict.get("label", label)
    telescope_name = args_dict["telescope"]
    label_with_telescope = f"{base_label}_{telescope_name}"

    calculator = IncidentAnglesCalculator(
        simtel_path=args_dict["simtel_path"],
        db_config=db_config,
        config_data={
            "telescope": args_dict["telescope"],
            "site": args_dict["site"],
            "model_version": args_dict["model_version"],
            "zenith_angle": args_dict["zenith"] * u.deg,
            "off_axis_angle": (
                (args_dict.get("off_axis_angles")[0] * u.deg)
                if args_dict.get("off_axis_angles")
                else 0.0 * u.deg
            ),
            "source_distance": args_dict["source_distance"] * u.km,
            "number_of_rays": int(args_dict.get("number_of_rays", 10000)),
        },
        output_dir=output_dir,
        label=base_label,
        perfect_mirror=bool(args_dict.get("perfect_mirror", False)),
        overwrite_rdna=bool(args_dict.get("overwrite_rdna", False)),
        mirror_reflection_random_angle=args_dict.get("mirror_reflection_random_angle", None),
        algn=args_dict.get("algn", None),
        test=args_dict.get("test", False),
        calculate_primary_secondary_angles=args_dict.get(
            "calculate_primary_secondary_angles", True
        ),
    )
    offsets = [float(v) for v in args_dict.get("off_axis_angles", [0.0])]

    results_by_offset = calculator.run_for_offsets(offsets)
    plot_incident_angles(results_by_offset, output_dir, label_with_telescope)
    total = sum(len(t) for t in results_by_offset.values())
    logger.info(
        f"Calculated incident angles for {len(results_by_offset)} offsets,\n"
        f"total photon statistics {total}"
    )


if __name__ == "__main__":
    main()

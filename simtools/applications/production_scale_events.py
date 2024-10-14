"""
Application to run the StatisticalErrorEvaluator.

Example Usage
-------------
1. Instantiate Evaluators:
    Create instances of `StatisticalErrorEvaluator` for each combination of
        zenith angles and offsets.

    ```python
    import os
    import numpy as np
    from simtools.production_configuration.calculate_statistical_errors_grid_point import (
        StatisticalErrorEvaluator, InterpolationHandler
    )

    base_path = "/path/"
    zeniths = [20, 40, 60]
    offsets = [0, 1, 2, 3, 4, 5]  # Offsets in degrees

    evaluator_instances = []
    for zenith in zeniths:
        for offset in offsets:
            file_name = (
                f"prod5b-LaPalma-{zenith}deg-lin51-LL/"
                f"gamma_onSource.N.BL-4LSTs15MSTs-MSTN_ID0.eff-{offset}-CUT0.fits"
                if offset == 0
                else f"prod5b-LaPalma-{zenith}deg-lin51-LL/"
                f"gamma_cone.N.BL-4LSTs15MSTs-MSTN_ID0.eff-{offset}-CUT0.fits"
            )
            file_path = os.path.join(base_path, file_name)
            offset_value = 2 * offset
            evaluator_instances.append(
                StatisticalErrorEvaluator(
                    file_path,
                    file_type="On-source" if offset == 0 else "Gamma-cone",
                    metrics={
                        "error_eff_area": 0.02,
                        "error_sig_eff_gh": 0.02,
                        "error_energy_estimate_bdt_reg_tree": 0.05,
                        "error_gamma_ray_psf": 0.01,
                        "error_image_template_methods": 0.03,
                    },
                    grid_point=(1, 180, zenith, 0, offset_value),
                )
            )
    ```

2. Calculate Metrics:
    For each evaluator instance, calculate metrics and scaled events.

    ```python
    for evaluator in evaluator_instances:
        evaluator.calculate_metrics()
        evaluator.calculate_scaled_events()
    ```

3. Interpolate Results:
    Create an InterpolationHandler and interpolate results for a specified grid point.

    ```python
    interpolation_handler = InterpolationHandler(evaluator_instances)

    grid_point = (1, 180, 30, 0, 0)  # Example grid point (energy, azimuth, zenith, NSB, offset)
    query_points = np.array([grid_point])
    scaled_events = interpolation_handler.interpolate(query_points)
    print(f"Scaled events for grid point {grid_point}: {scaled_events}")
    ```

from simtools.production_configuration.calculate_statistical_errors_grid_point import (
    StatisticalErrorEvaluator,
)
"""

import argparse
import logging
import os

import numpy as np

from simtools.production_configuration.calculate_statistical_errors_grid_point import (
    InterpolationHandler,
    StatisticalErrorEvaluator,
)


def _parse_command_line_arguments():
    """
    Parse command line arguments for the statistical error evaluator application.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate statistical errors from FITS files and interpolate results."
    )

    parser.add_argument(
        "--base_path", type=str, required=False, help="Path to the FITS files for interpolation."
    )
    parser.add_argument("--zeniths", nargs="+", type=int, help="List of zenith angles.")
    parser.add_argument("--offsets", nargs="+", type=int, help="List of offsets in degrees.")

    parser.add_argument(
        "--interpolate", action="store_true", help="Interpolate results for a specific grid point."
    )
    parser.add_argument(
        "--query_point",
        nargs=5,
        type=int,
        help="Grid point for interpolation (energy, azimuth, zenith, NSB, offset).",
    )

    return parser.parse_args()


def main():
    """Run the evaluator and interpolate."""
    args = _parse_command_line_arguments()

    logging.basicConfig(level=logging.INFO)

    evaluator_instances = []

    if args.base_path and args.zeniths and args.offsets:
        for zenith in args.zeniths:
            for offset in args.offsets:
                file_name = (
                    f"prod5b-LaPalma-{zenith}deg-lin51-LL/"
                    f"gamma_onSource.N.BL-4LSTs15MSTs-MSTN_ID0.eff-{offset}-CUT0.fits"
                    if offset == 0
                    else f"prod5b-LaPalma-{zenith}deg-lin51-LL/"
                    f"gamma_cone.N.BL-4LSTs15MSTs-MSTN_ID0.eff-{offset}-CUT0.fits"
                )
                file_path = os.path.join(args.base_path, file_name)
                offset_value = 2 * offset
                evaluator_instances.append(
                    StatisticalErrorEvaluator(
                        file_path,
                        file_type="On-source" if offset == 0 else "Gamma-cone",
                        metrics={
                            "error_eff_area": 0.02,
                            "error_sig_eff_gh": 0.02,
                            "error_energy_estimate_bdt_reg_tree": 0.05,
                            "error_gamma_ray_psf": 0.01,
                            "error_image_template_methods": 0.03,
                        },
                        grid_point=(1, 180, zenith, 0, offset_value),
                    )
                )

        # Calculate metrics and scaled events
        for evaluator in evaluator_instances:
            evaluator.calculate_metrics()
            evaluator.calculate_scaled_events()

    if args.interpolate and args.query_point:
        # Perform interpolation for the given query point
        interpolation_handler = InterpolationHandler(evaluator_instances)
        query_points = np.array([args.query_point])
        scaled_events = interpolation_handler.interpolate(query_points)
        print(f"Scaled events for grid point {args.query_point}: {scaled_events}")


if __name__ == "__main__":
    main()

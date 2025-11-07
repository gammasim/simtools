#!/usr/bin/env python3
r"""Derive Gaussian sigma and exponential tau from rise/fall widths.

This application solves for (sigma, tau) of a Gaussian convolved with a
causal exponential such that the resulting pulse matches the requested
rise and fall width definitions. It uses the solver implemented in
``simtools.simtel.pulse_shapes.solve_sigma_tau_from_risefall``.

Inputs:
- rise/fall widths and their fractional ranges
- optional dt and explicit time window

Outputs:
- Logs the derived parameters
- Writes a small JSON file with {"sigma_ns", "tau_ns"} in the output directory

Command line arguments
----------------------
- rise_width_ns (float, required)
    Target width on the rising edge in ns between rise_range fractions.
- fall_width_ns (float, required)
    Target width on the falling edge in ns between fall_range fractions.
- rise_range (float float, optional)
    Fractional amplitudes (low high) for rise width, e.g. 0.1 0.9 (default: 0.1 0.9).
- fall_range (float float, optional)
    Fractional amplitudes (high low) for fall width, e.g. 0.9 0.1 (default: 0.9 0.1).
- dt_ns (float, optional)
    Time sampling step in ns used by the solver (default: 0.1 ns).
- t_start_ns (float, optional)
    Explicit start time of the internal sampling window (ns) (default: -10 ns).
- t_stop_ns (float, optional)
    Explicit stop time of the internal sampling window (ns) (default: 25 ns).

Example
-------

Derive trigger rates for the South Alpha layout:

.. code-block:: console

    simtools-derive-pulse-shape-parameters \
      --rise_width_ns 2.0 \
      --fall_width_ns 6.0 \
      --rise_range 0.1 0.9 \
      --fall_range 0.9 0.1 \
      --dt_ns 0.1 \
      --t_start_ns -25 \
      --t_stop_ns 25
"""

import json
import logging
from pathlib import Path

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.simtel.pulse_shapes import solve_sigma_tau_from_rise_fall


def _parse():
    """Parse command line configuration for parameter derivation."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description=(
            "Derive Gaussian sigma and exponential tau from rise/fall width specifications."
        ),
    )

    config.parser.add_argument(
        "--rise_width_ns",
        help="Target width on the rising edge in ns between rise_range fractions.",
        type=float,
        required=True,
    )
    config.parser.add_argument(
        "--fall_width_ns",
        help="Target width on the falling edge in ns between fall_range fractions.",
        type=float,
        required=True,
    )
    config.parser.add_argument(
        "--rise_range",
        help="Fractional amplitudes (low high) for rise width, e.g. 0.1 0.9",
        type=float,
        nargs=2,
        default=[0.1, 0.9],
        required=False,
    )
    config.parser.add_argument(
        "--fall_range",
        help="Fractional amplitudes (high low) for fall width, e.g. 0.9 0.1",
        type=float,
        nargs=2,
        default=[0.9, 0.1],
        required=False,
    )
    config.parser.add_argument(
        "--dt_ns",
        help="Time sampling step in ns used by the solver.",
        type=float,
        default=0.1,
        required=False,
    )
    config.parser.add_argument(
        "--t_start_ns",
        help="Explicit start time of the internal sampling window (ns).",
        type=float,
        default=-10.0,
        required=False,
    )
    config.parser.add_argument(
        "--t_stop_ns",
        help="Explicit stop time of the internal sampling window (ns).",
        type=float,
        default=25.0,
        required=False,
    )

    return config.initialize()


def main():
    """Run parameter derivation and write results."""
    app_context = startup_application(_parse)
    log = logging.getLogger(__name__)

    rise_width_ns = float(app_context.args["rise_width_ns"])
    fall_width_ns = float(app_context.args["fall_width_ns"])
    rise_range = tuple(app_context.args["rise_range"])
    fall_range = tuple(app_context.args["fall_range"])
    dt_ns = float(app_context.args["dt_ns"])
    t_start_ns = float(app_context.args["t_start_ns"])
    t_stop_ns = float(app_context.args["t_stop_ns"])

    sigma_ns, tau_ns = solve_sigma_tau_from_rise_fall(
        rise_width_ns=rise_width_ns,
        fall_width_ns=fall_width_ns,
        dt_ns=dt_ns,
        rise_range=rise_range,
        t_start_ns=t_start_ns,
        t_stop_ns=t_stop_ns,
    )

    log.info(
        f"Derived pulse parameters: sigma={sigma_ns:.6g} ns, tau={tau_ns:.6g} ns "
        f"(rise={rise_width_ns} ns @ {rise_range}, fall={fall_width_ns} ns @ {fall_range})"
    )

    out_dir = app_context.io_handler.get_output_directory()
    base = app_context.args.get("label") or get_application_label(__file__)
    out_path = Path(out_dir) / f"{base}_pulse_parameters.json"

    parameters = {
        "sigma_ns": sigma_ns,
        "tau_ns": tau_ns,
        "inputs": {
            "rise_width_ns": rise_width_ns,
            "fall_width_ns": fall_width_ns,
            "rise_range": list(rise_range),
            "fall_range": list(fall_range),
            "dt_ns": dt_ns,
            "t_start_ns": t_start_ns,
            "t_stop_ns": t_stop_ns,
        },
    }
    out_path.write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    log.info(f"Wrote derived parameters to {out_path}")


if __name__ == "__main__":
    main()

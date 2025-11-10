#!/usr/bin/env python3
r"""Derive Gaussian sigma and exponential tau from specified rise/fall widths.

Solve (sigma, tau) for a Gaussian convolved with a causal exponential so the
pulse matches user-provided rise and fall widths between fractional amplitude
levels (e.g. 0.1-0.9 rise, 0.9-0.1 fall).

Command line arguments
----------------------
site (str, required)
        North or South.
telescope (str, required)
        Telescope model name.
model_version (str, required)
        Model version.
parameter_version (str, required)
    Parameter version.
rise_width_ns (float, required)
    Rising-edge width between rise_range fractions (ns).
fall_width_ns (float, required)
    Falling-edge width between fall_range fractions (ns).
rise_range (float float, optional)
    Fractional amplitudes (low high) for rise width (default: 0.1 0.9).
fall_range (float float, optional)
    Fractional amplitudes (high low) for fall width (default: 0.9 0.1).
dt_ns (float, optional)
    Time sampling step (ns). Default: 0.1.
time_margin_ns (float, optional)
    Margin added at both ends of readout window. Default: 5.


Example
-------
Derive parameters for a pulse with 2.5 ns rise (10-90%) and
 5 ns fall (90-10%) for LSTN-01:

.. code-block:: console

    simtools-derive-pulse-shape-parameters \
    --site North \
    --telescope MSTx-NectarCam \
    --model_version 7.0 \
    --parameter_version 1.0.0 \
    --rise_width_ns 2.5 \
    --fall_width_ns 5.0 \
    --rise_range 0.1 0.9 \
    --fall_range 0.9 0.1 \
    --dt_ns 0.1 \
    --time_margin_ns 10
"""

import logging

import simtools.data_model.model_data_writer as writer
from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.model.model_utils import initialize_simulation_models
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
        help="Wdth on the rising edge in ns between rise_range fractions.",
        type=float,
        required=True,
    )
    config.parser.add_argument(
        "--fall_width_ns",
        help="Width on the falling edge in ns between fall_range fractions.",
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
        "--time_margin_ns",
        help=(
            "Margin (ns) added to both ends of the instrument readout window when deriving the "
            "internal time window."
        ),
        type=float,
        default=10.0,
        required=False,
    )

    return config.initialize(
        db_config=True,
        simulation_model=["site", "telescope", "model_version", "parameter_version"],
        output=True,
    )


def main():
    """Run parameter derivation and write results."""
    app_context = startup_application(_parse)
    log = logging.getLogger(__name__)

    rise_width_ns = app_context.args["rise_width_ns"]
    fall_width_ns = app_context.args["fall_width_ns"]
    rise_range = tuple(app_context.args["rise_range"])
    fall_range = tuple(app_context.args["fall_range"])
    dt_ns = app_context.args["dt_ns"]
    time_margin_ns = app_context.args["time_margin_ns"]
    site = app_context.args["site"]
    label = app_context.args.get("label") or get_application_label(__file__)
    telescope_model, _, _ = initialize_simulation_models(
        label=label,
        db_config=app_context.db_config,
        model_version=app_context.args["model_version"],
        site=site,
        telescope_name=app_context.args["telescope"],
    )
    fadc_sum_bins = telescope_model.get_parameter_value("fadc_sum_bins")

    window_ns = fadc_sum_bins + time_margin_ns
    t_start_ns = -window_ns
    t_stop_ns = window_ns

    sigma_ns, tau_ns = solve_sigma_tau_from_rise_fall(
        rise_width_ns=rise_width_ns,
        fall_width_ns=fall_width_ns,
        dt_ns=dt_ns,
        rise_range=rise_range,
        t_start_ns=t_start_ns,
        t_stop_ns=t_stop_ns,
    )
    # Apply reasonable rounding for output precision.
    sigma_ns = round(sigma_ns, 4)
    tau_ns = round(tau_ns, 4)

    log.info(
        f"Derived pulse parameters: sigma={sigma_ns:.6g} ns, tau={tau_ns:.6g} ns "
        f"(rise={rise_width_ns} ns @ {rise_range}, fall={fall_width_ns} ns @ {fall_range})"
    )

    output_path = app_context.args.get("output_path")
    instrument = app_context.args.get("telescope")
    parameter_version = app_context.args.get("parameter_version")

    writer.ModelDataWriter.dump_model_parameter(
        parameter_name="flasher_pulse_width",
        value=sigma_ns,
        instrument=instrument,
        parameter_version=parameter_version,
        output_file="flasher_pulse_width.json",
        output_path=output_path,
        unit="ns",
    )
    writer.ModelDataWriter.dump_model_parameter(
        parameter_name="flasher_pulse_exp_decay",
        value=tau_ns,
        instrument=instrument,
        parameter_version=parameter_version,
        output_file="flasher_pulse_exp_decay.json",
        output_path=output_path,
        unit="ns",
    )
    log.info(
        f"Wrote model parameter files flasher_pulse_width.json and "
        f"flasher_pulse_exp_decay.json (sigma={sigma_ns:.6g} ns, tau={tau_ns:.6g} ns)"
    )


if __name__ == "__main__":
    main()

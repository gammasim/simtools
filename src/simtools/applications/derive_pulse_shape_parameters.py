#!/usr/bin/env python3
"""Derive Gaussian sigma and exponential tau from specified rise/fall widths."""

import logging

import simtools.data_model.model_data_writer as writer
from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.model.model_utils import initialize_simulation_models
from simtools.simtel.pulse_shapes import solve_sigma_tau_from_rise_fall

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "rise_width_ns",
        help="Width on the rising edge in ns between rise_range fractions.",
        type=float,
        required=True,
    ),
    cli.ArgumentDefinition(
        "fall_width_ns",
        help="Width on the falling edge in ns between fall_range fractions.",
        type=float,
        required=True,
    ),
    cli.ArgumentDefinition(
        "rise_range",
        help="Fractional amplitudes (low high) for rise width, e.g. 0.1 0.9",
        type=float,
        nargs=2,
        default=[0.1, 0.9],
        required=False,
    ),
    cli.ArgumentDefinition(
        "fall_range",
        help="Fractional amplitudes (high low) for fall width, e.g. 0.9 0.1",
        type=float,
        nargs=2,
        default=[0.9, 0.1],
        required=False,
    ),
    cli.ArgumentDefinition(
        "dt_ns",
        help="Time sampling step in ns used by the solver.",
        type=float,
        default=0.1,
        required=False,
    ),
    cli.ArgumentDefinition(
        "time_margin_ns",
        help=(
            "Margin (ns) added to both ends of the instrument readout window when deriving "
            "the internal time window."
        ),
        type=float,
        default=10.0,
        required=False,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.PARAMETER_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        cli.TELESCOPE,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    database=True,
    initialize_output=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    log = logging.getLogger(__name__)

    rise_width_ns = app_context.args["rise_width_ns"]
    fall_width_ns = app_context.args["fall_width_ns"]
    rise_range = tuple(app_context.args["rise_range"])
    fall_range = tuple(app_context.args["fall_range"])
    dt_ns = app_context.args["dt_ns"]
    time_margin_ns = app_context.args["time_margin_ns"]
    site = app_context.args["site"]
    label = app_context.args.get("label") or app_context.args["application_label"]
    telescope_model, _, _ = initialize_simulation_models(
        label=label,
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

    writer.ModelDataWriter.write_model_parameter(
        parameter_name="flasher_pulse_width",
        value=sigma_ns,
        instrument=instrument,
        parameter_version=parameter_version,
        output_file="flasher_pulse_width.json",
        output_path=output_path,
        unit="ns",
    )
    writer.ModelDataWriter.write_model_parameter(
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

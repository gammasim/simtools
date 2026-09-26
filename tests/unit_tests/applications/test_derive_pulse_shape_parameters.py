from unittest.mock import Mock, patch

import astropy.units as u
import pytest


@patch("simtools.applications.derive_pulse_shape_parameters.writer.ModelDataWriter")
@patch("simtools.applications.derive_pulse_shape_parameters.solve_sigma_tau_from_rise_fall")
@patch("simtools.applications.derive_pulse_shape_parameters.initialize_simulation_models")
@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_converts_fadc_bins_to_nanoseconds(
    mock_application_start, mock_initialize_models, mock_solver, mock_writer
):
    from simtools.applications.derive_pulse_shape_parameters import main

    context = Mock()
    context.args = {
        "rise_width_ns": 2.5,
        "fall_width_ns": 5.0,
        "rise_range": [0.1, 0.9],
        "fall_range": [0.9, 0.1],
        "dt_ns": 0.1,
        "time_margin_ns": 5.0,
        "site": "North",
        "telescope": "MSTN-01",
        "model_version": "7.0.0",
        "parameter_version": "1.0.0",
        "application_label": "test",
        "output_path": None,
    }
    mock_application_start.return_value = context
    telescope_model = Mock()
    telescope_model.get_parameter_value.return_value = 40
    telescope_model.get_parameter_value_with_unit.return_value = 250 * u.MHz
    mock_initialize_models.return_value = (telescope_model, Mock(), Mock())
    mock_solver.return_value = (1.0, 2.0)

    main()

    call = mock_solver.call_args.kwargs
    assert call["t_start_ns"] == -165.0
    assert call["t_stop_ns"] == 165.0


@patch("simtools.applications.derive_pulse_shape_parameters.solve_sigma_tau_from_rise_fall")
@patch("simtools.applications.derive_pulse_shape_parameters.initialize_simulation_models")
@patch("simtools.application.definition.ApplicationDefinition.start")
@pytest.mark.parametrize("fadc_mhz", [0.0, -1.0, float("nan"), float("inf")])
def test_main_rejects_invalid_fadc_rate(
    mock_application_start, mock_initialize_models, mock_solver, fadc_mhz
):
    from simtools.applications.derive_pulse_shape_parameters import main

    context = Mock()
    context.args = {
        "rise_width_ns": 2.5,
        "fall_width_ns": 5.0,
        "rise_range": [0.1, 0.9],
        "fall_range": [0.9, 0.1],
        "dt_ns": 0.1,
        "time_margin_ns": 5.0,
        "site": "North",
        "telescope": "MSTN-01",
        "model_version": "7.0.0",
        "parameter_version": "1.0.0",
        "application_label": "test",
    }
    mock_application_start.return_value = context
    telescope_model = Mock()
    telescope_model.get_parameter_value.return_value = 40
    telescope_model.get_parameter_value_with_unit.return_value = fadc_mhz * u.MHz
    mock_initialize_models.return_value = (telescope_model, Mock(), Mock())

    with pytest.raises(ValueError, match="finite and positive"):
        main()

    mock_solver.assert_not_called()

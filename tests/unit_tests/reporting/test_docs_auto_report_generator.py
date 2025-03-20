from unittest.mock import MagicMock, call, patch

import pytest

from simtools.reporting.docs_auto_report_generator import ReportGenerator


def test_add_design_models_to_telescopes(io_handler, db_config):
    args = {"site": "North", "telescope": "LSTN-01"}
    output_path = io_handler.get_output_directory()
    report_generator = ReportGenerator(db_config, args, output_path)

    telescopes = ["LSTN-01", "LSTN-02"]
    model_version = "6.0.0"

    # Mocking the design model responses
    report_generator.db.get_design_model = MagicMock()
    report_generator.db.get_design_model.side_effect = ["LSTN-design", None]

    result = report_generator._add_design_models_to_telescopes(model_version, telescopes)

    assert len(result) == 3  # Original telescopes + one design model
    assert "LSTN-design" in result
    assert "LSTN-01" in result
    assert "LSTN-02" in result


def test_filter_telescopes_by_site(io_handler, db_config):
    args = {"site": "North", "telescope": "LSTN-01"}
    output_path = io_handler.get_output_directory()
    report_generator = ReportGenerator(db_config, args, output_path)

    telescopes = ["LSTN-01", "LSTN-02", "MSTS-01", "MSTN-01"]
    selected_sites = {"North"}

    with patch("simtools.utils.names.get_site_from_array_element_name") as mock_get_site:
        mock_get_site.side_effect = [["North"], ["North"], ["South"], ["North"]]
        result = report_generator._filter_telescopes_by_site(telescopes, selected_sites)

    assert len(result) == 3
    assert "MSTS-01" not in result
    assert all(tel in result for tel in ["LSTN-01", "LSTN-02", "MSTN-01"])


@pytest.mark.parametrize(
    ("all_flags", "expected_count"),
    [
        ({"all_telescopes": True, "all_sites": True, "all_model_versions": True}, 6),
        ({"all_telescopes": False, "all_sites": False, "all_model_versions": False}, 1),
    ],
)
def test_get_report_parameters(io_handler, db_config, all_flags, expected_count):
    args = {"site": "North", "telescope": "LSTN-01", "model_version": "6.0.0", **all_flags}
    output_path = io_handler.get_output_directory()
    report_generator = ReportGenerator(db_config, args, output_path)

    with patch.multiple(
        report_generator.db,
        get_model_versions=MagicMock(return_value=["5.0.0", "6.0.0"]),
        get_array_elements=MagicMock(return_value=["LSTN-01", "LSTN-02", "LSTN-design"]),
    ):
        with patch("simtools.utils.names.get_site_from_array_element_name", return_value=["North"]):
            result = list(report_generator._get_report_parameters())

    assert len(result) == expected_count
    for version, telescope, site in result:
        assert version in ["5.0.0", "6.0.0"]
        assert telescope in ["LSTN-01", "LSTN-02", "LSTN-design"]
        assert site == "North"


def test_get_telescopes_from_layout(io_handler, db_config):
    args = {"all_telescopes": True, "site": "North"}
    output_path = io_handler.get_output_directory()
    report_generator = ReportGenerator(db_config, args, output_path)

    mock_layouts = {
        "6.0.0": {
            "array_layouts": {
                "value": [
                    {"name": "hyper_array", "elements": ["LSTN-01", "LSTN-02"]},
                    {"name": "other_layout", "elements": ["MSTN-01"]},
                ]
            }
        }
    }

    with patch.multiple(
        report_generator.db,
        get_model_parameters_for_all_model_versions=MagicMock(return_value=mock_layouts),
        get_design_model=MagicMock(return_value="LSTN-design"),
    ):
        result = report_generator._get_telescopes_from_layout("North")

    assert len(result) == 3  # LSTN-01, LSTN-02, LSTN-design
    assert all(tel in result for tel in ["LSTN-01", "LSTN-02", "LSTN-design"])


def test_generate_parameter_report_combinations(io_handler, db_config):
    args = {"all_telescopes": True, "all_sites": True}
    output_path = io_handler.get_output_directory()
    report_generator = ReportGenerator(db_config, args, output_path)

    mock_telescopes = {"LSTN-01", "LSTN-02", "LSTN-design"}

    # Mock both the site selection and telescope layout retrieval
    with (
        patch(
            "simtools.utils.names.get_site_from_array_element_name",
            side_effect=lambda x: ["North"] if x[3] == "N" else ["South"],
        ),
        patch.multiple(
            report_generator,
            _get_telescopes_from_layout=MagicMock(
                side_effect=lambda site: mock_telescopes if site == "North" else set()
            ),
        ),
    ):
        result = list(report_generator._generate_parameter_report_combinations())

    expected_combinations = {("LSTN-01", "North"), ("LSTN-02", "North"), ("LSTN-design", "North")}
    assert set(result) == expected_combinations
    assert len(result) == 3


def test_auto_generate_array_element_reports(io_handler, db_config):
    args = {"all_telescopes": True, "all_sites": True, "all_model_versions": True}
    output_path = io_handler.get_output_directory()
    report_generator = ReportGenerator(db_config, args, output_path)

    # Mock parameters that would be returned for different sites and telescopes
    mock_params = [
        ("6.0.0", "LSTN-01", "North"),
        ("6.0.0", "LSTN-02", "North"),
        ("6.0.0", "MSTN-01", "North"),
        ("6.0.0", "MSTS-01", "South"),
        ("6.0.0", "MSTS-02", "South"),
        ("5.0.0", "LSTN-01", "North"),
        ("5.0.0", "LSTN-02", "North"),
        ("5.0.0", "MSTN-01", "North"),
        ("5.0.0", "MSTS-01", "South"),
        ("5.0.0", "MSTS-02", "South"),
    ]

    mock_report_gen = MagicMock()

    with (
        patch.object(report_generator, "_get_report_parameters", return_value=mock_params),
        patch.object(
            report_generator, "_generate_single_array_element_report", side_effect=mock_report_gen
        ),
    ):
        report_generator.auto_generate_array_element_reports()

        # Verify that _generate_single_array_element_report was called for each combination
        assert mock_report_gen.call_count == 10
        mock_report_gen.assert_has_calls(
            [
                call("6.0.0", "LSTN-01", "North"),
                call("6.0.0", "LSTN-02", "North"),
                call("6.0.0", "MSTN-01", "North"),
                call("6.0.0", "MSTS-01", "South"),
                call("6.0.0", "MSTS-02", "South"),
                call("5.0.0", "LSTN-01", "North"),
                call("5.0.0", "LSTN-02", "North"),
                call("5.0.0", "MSTN-01", "North"),
                call("5.0.0", "MSTS-01", "South"),
                call("5.0.0", "MSTS-02", "South"),
            ],
            any_order=True,
        )


def test_auto_generate_parameter_reports(io_handler, db_config):
    """Test parameter report generation for multiple telescopes and sites."""
    args = {"all_telescopes": True, "all_sites": True}
    output_path = io_handler.get_output_directory()
    report_generator = ReportGenerator(db_config, args, output_path)

    # Mock telescope-site combinations that would be returned
    mock_combinations = [
        ("LSTN-01", "North"),
        ("LSTN-02", "North"),
        ("MSTS-01", "South"),
        ("MSTS-02", "South"),
    ]

    with (
        patch.object(
            report_generator,
            "_generate_parameter_report_combinations",
            return_value=mock_combinations,
        ),
        patch(
            "simtools.reporting.docs_read_parameters.ReadParameters.produce_model_parameter_reports",
            new_callable=MagicMock,
        ) as mock_produce,
    ):
        report_generator.auto_generate_parameter_reports()

        # Verify that produce_model_parameter_reports was called for each combination
        assert mock_produce.call_count == len(mock_combinations)

        # Check that each telescope-site combination was processed with correct args
        expected_calls = [call() for _ in mock_combinations]
        mock_produce.assert_has_calls(expected_calls, any_order=True)

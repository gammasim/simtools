from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from simtools.reporting.docs_auto_report_generator import ReportGenerator


def test__add_design_models_to_telescopes(io_handler, db_config):
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


def test__filter_telescopes_by_site(io_handler, db_config):
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
def test__get_report_parameters(io_handler, db_config, all_flags, expected_count):
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


def test__get_telescopes_from_layout(io_handler, db_config):
    """Test getting telescopes from layout for both all_telescopes=True and False cases."""
    test_cases = [
        # Test case 1: all_telescopes=True
        {
            "args": {"all_telescopes": True, "site": "North"},
            "mock_layouts": {
                "6.0.0": {
                    "array_layouts": {
                        "value": [
                            {"name": "hyper_array", "elements": ["LSTN-01", "LSTN-02"]},
                            {"name": "other_layout", "elements": ["MSTN-01"]},
                        ]
                    }
                }
            },
            "mock_design_model": "LSTN-design",
            "expected_telescopes": {"LSTN-01", "LSTN-02", "LSTN-design"},
        },
        # Test case 2: all_telescopes=False
        {
            "args": {"all_telescopes": False, "site": "North", "telescope": "LSTN-01"},
            "mock_layouts": {},  # Not used in this case
            "mock_design_model": None,  # Not used in this case
            "expected_telescopes": {"LSTN-01"},
        },
    ]

    for case in test_cases:
        output_path = io_handler.get_output_directory()
        report_generator = ReportGenerator(db_config, case["args"], output_path)

        with patch.multiple(
            report_generator.db,
            get_model_parameters_for_all_model_versions=MagicMock(
                return_value=case["mock_layouts"]
            ),
            get_design_model=MagicMock(return_value=case["mock_design_model"]),
        ):
            result = report_generator._get_telescopes_from_layout("North")

            # Verify the result
            assert result == case["expected_telescopes"]

            # Verify DB methods were called correctly based on all_telescopes flag
            if case["args"]["all_telescopes"]:
                report_generator.db.get_model_parameters_for_all_model_versions.assert_called_once()
                assert len(result) == len(case["expected_telescopes"])
            else:
                # Verify DB methods were not called when all_telescopes=False
                report_generator.db.get_model_parameters_for_all_model_versions.assert_not_called()
                report_generator.db.get_design_model.assert_not_called()
                assert result == {case["args"]["telescope"]}


def test__generate_parameter_report_combinations(io_handler, db_config):
    """Test parameter report combinations generation for both specific and all telescopes."""

    test_cases = [
        # Test case 1: Specific telescope (all_telescopes=False)
        {
            "args": {"all_telescopes": False, "all_sites": True, "telescope": "LSTN-01"},
            "mock_valid_sites": ["North"],
            "expected_combinations": [("LSTN-01", "North")],
        },
        # Test case 2: All telescopes
        {
            "args": {"all_telescopes": True, "all_sites": True},
            "mock_telescopes_north": {"LSTN-01", "LSTN-02", "LSTN-design"},
            "mock_telescopes_south": {"MSTS-01", "MSTS-02"},
            "expected_combinations": [
                ("LSTN-01", "North"),
                ("LSTN-02", "North"),
                ("LSTN-design", "North"),
                ("MSTS-01", "South"),
                ("MSTS-02", "South"),
            ],
        },
    ]

    for case in test_cases:
        output_path = io_handler.get_output_directory()
        report_generator = ReportGenerator(db_config, case["args"], output_path)

        with (
            patch(
                "simtools.utils.names.get_site_from_array_element_name",
                side_effect=lambda x: ["North"]
                if x[3] == "N"
                else ["South"]
                if not case["args"].get("all_telescopes")
                else case.get("mock_valid_sites", []),
            ),
            patch.multiple(
                report_generator,
                _get_telescopes_from_layout=MagicMock(
                    side_effect=lambda site: (
                        case.get("mock_telescopes_north", set())
                        if site == "North"
                        else case.get("mock_telescopes_south", set())
                    )
                ),
            ),
        ):
            result = list(report_generator._generate_parameter_report_combinations())
            expected = case["expected_combinations"]

            # Verify combinations
            assert len(result) == len(expected)
            assert set(result) == set(expected)

            # For specific telescope case, verify _get_valid_sites_for_telescope was used
            if not case["args"].get("all_telescopes"):
                assert all(combo[0] == case["args"]["telescope"] for combo in result)


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


def test__generate_single_array_element_report(io_handler, db_config):
    """Test generation of a single array element report."""
    # Initialize ReportGenerator with basic args
    args = {"all_telescopes": True, "all_sites": True}
    output_path = io_handler.get_output_directory()
    report_generator = ReportGenerator(db_config, args, output_path)

    # Test parameters
    model_version = "6.0.0"
    telescope = "LSTN-01"
    site = "North"

    # Mock ReadParameters and its produce_array_element_report method
    mock_read_params = MagicMock()

    with patch(
        "simtools.reporting.docs_auto_report_generator.ReadParameters",
        return_value=mock_read_params,
    ) as mock_read_params_class:
        # Call the method under test
        report_generator._generate_single_array_element_report(model_version, telescope, site)

        # Verify ReadParameters was instantiated with correct arguments
        expected_args = {
            "all_telescopes": True,
            "all_sites": True,
            "telescope": telescope,
            "site": site,
            "model_version": model_version,
        }
        expected_output_path = Path(output_path) / str(model_version)

        mock_read_params_class.assert_called_once_with(
            db_config, expected_args, expected_output_path
        )

        # Verify produce_array_element_report was called
        mock_read_params.produce_array_element_report.assert_called_once()


def test__get_valid_sites_for_telescope(io_handler, db_config):
    """Test getting valid sites for different telescope types."""
    args = {"all_telescopes": True, "all_sites": True}
    output_path = io_handler.get_output_directory()
    report_generator = ReportGenerator(db_config, args, output_path)

    test_cases = [
        # Test single site telescope (LST North)
        {"telescope": "LSTN-01", "mock_return": "North", "expected": ["North"]},
        # Test multi-site telescope (MST)
        {
            "telescope": "MST-FlashCam",
            "mock_return": ["North", "South"],
            "expected": ["North", "South"],
        },
        # Test South site telescope
        {"telescope": "MSTS-01", "mock_return": "South", "expected": ["South"]},
    ]

    for case in test_cases:
        with patch(
            "simtools.utils.names.get_site_from_array_element_name",
            return_value=case["mock_return"],
        ) as mock_get_site:
            result = report_generator._get_valid_sites_for_telescope(case["telescope"])

            # Verify get_site_from_array_element_name was called with correct telescope
            mock_get_site.assert_called_once_with(case["telescope"])

            # Verify returned sites match expected
            assert result == case["expected"]

            # Verify result is always a list
            assert isinstance(result, list)

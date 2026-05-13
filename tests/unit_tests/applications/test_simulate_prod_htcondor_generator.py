#!/usr/bin/python3

from types import SimpleNamespace
from unittest.mock import patch

import simtools.applications.simulate_prod_htcondor_generator as app


@patch(
    "simtools.applications.simulate_prod_htcondor_generator.htcondor_script_generator.generate_submission_script"
)
@patch("simtools.applications.simulate_prod_htcondor_generator.build_application")
def test_main_uses_standard_build_application(
    mock_build_application, mock_generate_submission_script
):
    mock_build_application.return_value = SimpleNamespace(args={"output_path": "htcondor_submit"})

    app.main()

    assert mock_build_application.call_args.kwargs["initialization_kwargs"] == {
        "db_config": False,
        "preserve_by_version_keys": ["array_layout_name"],
        "simulation_model": ["site", "layout", "telescope", "model_version"],
        "simulation_configuration": {"software": None, "corsika_configuration": ["all"]},
    }
    mock_generate_submission_script.assert_called_once_with({"output_path": "htcondor_submit"})

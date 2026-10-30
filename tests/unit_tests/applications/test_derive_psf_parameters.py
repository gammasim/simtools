"""Tests for PSF parameter optimization backend dispatch."""

import pytest

from simtools.applications import derive_psf_parameters as app
from simtools.configuration import arguments as cli
from simtools.configuration.commandline_parser import CommandLineParser
from simtools.ray_tracing import psf_parameter_optimisation as psf_opt


@pytest.mark.parametrize("workers", [None, 1, 4])
def test_worker_argument(workers):
    parser = CommandLineParser()
    parser.add_argument_definitions(app._ARGUMENTS)
    arguments = [] if workers is None else ["--max_workers", str(workers)]
    assert parser.parse_args(arguments).max_workers == workers


def test_main_keeps_controller_local(mocker, tmp_test_directory, run_context):
    args = {
        "site": "North",
        "telescope": "LSTN-01",
        "model_version": "6.0.0",
        "backend": "htcondor",
        "backend_config": "backend.yml",
        "max_workers": 4,
    }
    context = run_context.replace(args=args)
    mocker.patch.object(type(app.APPLICATION), "start", return_value=context)
    workflow = mocker.patch.object(psf_opt, "run_psf_optimization_job")

    app.main()

    workflow.assert_called_once_with(
        psf_opt.PSFOptimizationJob(args, context.paths.get_output_directory(), context)
    )
    assert workflow.call_args.args[0].args is not args
    assert all(argument in app.APPLICATION.arguments for argument in cli.BACKEND_ARGUMENTS)

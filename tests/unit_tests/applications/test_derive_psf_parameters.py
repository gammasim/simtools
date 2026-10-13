"""Tests for PSF parameter optimization backend dispatch."""

import pickle
from types import SimpleNamespace

import pytest

from simtools.applications import derive_psf_parameters as app
from simtools.configuration import arguments as cli
from simtools.job_execution.backends.base import BackendExecutionError
from simtools.ray_tracing import psf_parameter_optimisation as psf_opt


@pytest.mark.parametrize("backend", [None, "local", "htcondor"])
def test_main_dispatches_workflow(mocker, tmp_test_directory, backend):
    args = {"site": "North", "telescope": "LSTN-01", "model_version": "6.0.0"}
    if backend is not None:
        args.update(backend=backend, backend_config="backend.yml")
    context = SimpleNamespace(
        args=args,
        io_handler=SimpleNamespace(get_output_directory=lambda: tmp_test_directory),
    )
    mocker.patch.object(type(app.APPLICATION), "start", return_value=context)
    dispatch = mocker.patch.object(app, "map_ordered")

    app.main()

    job = psf_opt.PSFOptimizationJob(args, tmp_test_directory)
    dispatch.assert_called_once_with(
        psf_opt.run_psf_optimization_job,
        [job],
        backend=backend or "local",
        backend_config=args.get("backend_config"),
    )
    submitted_job = dispatch.call_args.args[1][0]
    assert submitted_job.args is not args
    assert pickle.loads(pickle.dumps(submitted_job)) == job
    assert all(argument in app.APPLICATION.arguments for argument in cli.BACKEND_ARGUMENTS)


@pytest.mark.parametrize("fails", [False, True])
def test_main_local_execution(mocker, tmp_test_directory, fails):
    args = {"site": "North", "telescope": "LSTN-01", "model_version": "6.0.0"}
    context = SimpleNamespace(
        args=args,
        io_handler=SimpleNamespace(get_output_directory=lambda: tmp_test_directory),
    )
    mocker.patch.object(type(app.APPLICATION), "start", return_value=context)
    telescope, site = object(), object()
    mocker.patch.object(
        psf_opt, "initialize_simulation_models", return_value=(telescope, site, None)
    )
    workflow = mocker.patch.object(psf_opt, "run_psf_optimization_workflow")
    if fails:
        workflow.side_effect = ValueError("Invalid PSF data")
        with pytest.raises(BackendExecutionError, match="Invalid PSF data"):
            app.main()
    else:
        app.main()
    workflow.assert_called_once_with(telescope, site, args, tmp_test_directory)

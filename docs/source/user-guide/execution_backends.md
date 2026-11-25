# Execution backends

Compute-intensive applications can be executed in parallel, either locally or on backends
like the HTCondor system.
The backend is selected with `--backend` and configured with `--backend_config`.

Available backends:

- `local` (default) uses `--max_workers` processes on the controller host.
- `htcondor` submits one scheduler process per job; `max_workers` is not a queue limit.


## HTCondor installation and runtime

Install the optional Python bindings on the submission host:

```bash
pip install "gammasimtools[htcondor]"
```

HTCondor uses a shared filesystem: payloads, outputs, logs, containers, and environment files
must be visible at the same absolute paths on the submit and execute hosts. Execute hosts need a
compatible Python environment and the dependencies required by the job or configured container.
Worker jobs support commands and importable Python functions. Job inputs must be trusted because
the worker imports and executes the specified function. Job data and results
support JSON values, paths, numeric NumPy arrays, quantities, and registered simtools records;
arbitrary Python objects and initializers are not supported.

The submission work directory and its ancestors must be owned by the submitting user or root
and must not be writable by other users, except for sticky directories. Worker identities need
group read and search access to job inputs and group write access to result directories. Workers
must not have the submission user's identity or additional write access to the inputs.
All workers with access to a result directory must be trusted: they can replace other jobs'
results, and result files do not authenticate their originating job.

## Backend configuration

Pass a YAML mapping with `--backend_config`:

```yaml
request_cpus: 1
priority: 0
request_memory: 4GB
request_disk: 10GB
container_image: /shared/containers/simtools.sif
# Keep the HTCondor scratch mount away from the image's /workdir environment.
container_target_dir: /simtools-run
environment_file: /shared/config/simtools.env
log_path: scheduler.log
poll_interval: 60
timeout: null
cancel_on_interrupt: false
keep_successful_artifacts: false
extra_submit_attributes: {}
```

For container jobs, the backend runs `python` inside the image by default. Set
`python_executable` only when the image uses a different command, such as `python3` or
`/opt/conda/bin/python`.

`poll_interval` is used only when an application waits for submitted jobs, for example with
`simtools-simulate-prod --wait`.

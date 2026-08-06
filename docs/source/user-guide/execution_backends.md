# Execution backends

Compute intensive application can be execution in parallel, either locally or on backends
like the HTCondor system.
The backend is selected with `--backend` and configured with `--backend_config`.

The following backends are available:

- Local execution `local` backend uses `--max_workers` processes on the controller host (default)
- HTCondor is selected with the `htcondor`. It submits one scheduler process per
job and does not use `max_workers` as a queue limit.


## HTCondor installation and runtime

Install the optional Python bindings on the submission host:

```bash
pip install "gammasimtools[htcondor]"
```

The initial backend uses a shared filesystem. Input payloads, application outputs, logs,
containers, and environment files must be visible at the same absolute paths on the submit and
execute hosts must provide a compatible Python environment and the dependencies required by the
serialized job, either directly or through the configured container.

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
# Python command available inside the container; use an absolute path if needed.
python_executable: python
environment_file: /shared/config/simtools.env
log_path: scheduler.log
poll_interval: 60
timeout: null
cancel_on_interrupt: false
keep_successful_artifacts: false
extra_submit_attributes: {}
```

## Manifests, logs, and failures

Each submission creates a private run directory containing `submission.json`, per-job inputs and
results, scheduler output streams, application logs, and one scheduler event log. The manifest
records the cluster and process mapping, job order, expected outputs, resource requests, and the
current controller state. It can be loaded with
{func}`job_execution.execution.load_submission` and passed to
{func}`job_execution.execution.wait_for_submission` after a detached controller exits.

The controller tracks the shared event log until every process terminates or the timeout expires.
Evicted jobs remain active because HTCondor may restart them. Held, removed, aborted, signalled,
nonzero, missing-result, and missing-output jobs are reported together with their process IDs and
diagnostic paths.

On interruption, jobs remain queued by default and the manifest is marked `interrupted`. Set
`cancel_on_interrupt: true` to remove the cluster instead. Failed runs retain all artifacts. After
a successful run, serialized inputs, results, stdout, and stderr are removed unless
`keep_successful_artifacts` is enabled; the manifest, scheduler event log, and application logs are
retained.

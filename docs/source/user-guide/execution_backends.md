# Execution backends

Applications that split work into independent jobs accept `--backend` and
`--backend_config`. The default `local` backend uses worker processes on the controller host;
`max_workers` affects only this backend. The `htcondor` backend submits one scheduler process per
job and does not use `max_workers` as a queue limit.

## HTCondor installation and runtime

Install the optional Python bindings on the submission host:

```bash
pip install "gammasimtools[htcondor]"
```

The initial backend uses a shared filesystem. Input payloads, application outputs, logs,
containers, and environment files must be visible at the same absolute paths on the submit and
execute hosts. Execute nodes must provide the same Python and simtools versions as the submission
host, either directly or through the configured container.

Worker payloads use Python pickle and are executable data. Keep the backend work directory private
and load submission manifests only from trusted runs created by simtools.

## Backend configuration

Pass a YAML mapping with `--backend_config`:

```yaml
request_cpus: 1
priority: 0
request_memory: 4GB
request_disk: 10GB
container_image: /shared/containers/simtools.sif
environment_file: /shared/config/simtools.env
log_path: scheduler.log
poll_interval: 60
timeout: null
cancel_on_interrupt: false
keep_successful_artifacts: false
extra_submit_attributes: {}
```

Unknown keys and protected scheduler fields fail before submission. The environment file accepts
`KEY=VALUE` entries; its contents are passed to HTCondor but are not copied into the durable
manifest or simtools logs.

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

# Run Productions

The basic execution unit of a production is one simulation run. It is executed with
[simtools-simulate-prod](../applications/simtools-simulate-prod). Large productions are built
from many such runs, usually generated from the ECSV production grid described in
[Configure Productions](configure-productions.md).

## Run a Single Production Job

`simtools-simulate-prod` prepares the simulation configuration, writes run scripts, executes the
selected simulation software, validates the produced files, and reports the result.

The selected `simulation_software` controls which part of the chain is executed:

- `corsika`: run shower simulations only
- `sim_telarray`: run detector simulation using an existing CORSIKA input file
- `corsika_sim_telarray`: run the CORSIKA to sim_telarray chain using multipipe execution

Typical outputs are:

- CORSIKA, sim_telarray, and simtools log files
- sim_telarray eventio files and histogram files
- reduced event-data HDF5 files (enabled by default; disable with `--reduced_event_lists`)
- output and log file lists when `--save_file_lists` is used
- copied grid output files, including logs, histogram files, model archives, moved event files,
  and `simtools.log.gz`, when `--grid_output_path` is configured

Example command:

```bash
simtools-simulate-prod \
    --model_version 7.0.0 \
    --site North \
    --array_layout_name LSTN-01 \
    --primary gamma \
    --azimuth_angle North \
    --zenith_angle 20 deg \
    --energy_range 30 GeV 300 GeV \
    --core_scatter 10 500 m \
    --view_cone 0 deg 10 deg \
    --showers_per_run 5 \
    --corsika_he_interaction qgs3 \
    --corsika_le_interaction urqmd \
    --corsika_hadronic_transition_energy 80 GeV \
    --run_number 1
```

Alternatively, a single job can be selected from an executable production grid. In this mode,
the selected row defines the production parameters such as primary, direction, energy range,
layout, model version, hadronic models and any explicitly configured transition energy, site, run
number, and simulation software. Do not combine
`--job_grid_file` with manual production arguments such as `--zenith_angle`; use only operational
options such as labels and output paths alongside the grid selection.

```bash
simtools-simulate-prod \
    --job_grid_file production_grid_points_horizontal.ecsv \
    --job_grid_row 1 \
    --label test \
    --output_path simtools-output
```

Example integration configurations are available in `tests/integration_tests/config`, including
`simulate_prod_gamma_40_deg_south_corsika_only.yml`,
`simulate_prod_gamma_40_deg_south_sim_telarray_only.yml`,
`simulate_prod_gamma_62_deg_south_check_output.yml`, and
`simulate_prod_proton_20_deg_north_check_output.yml`.

## Running simtools on HTCondor using Apptainers

Large productions can be submitted directly from `simtools-simulate-prod` using the generic
HTCondor backend. The application reads the ECSV job grid written by
[simtools-production-generate-grid](../applications/simtools-production-generate-grid), submits one
job per row, waits for completion, and writes scheduler logs. See
[Execution backends](../execution_backends.md) for runtime assumptions, configuration details,
manifest recovery, and failure behavior.

Install the optional scheduler extra on the submission host:

```bash
pip install "gammasimtools[htcondor]"
```

Apptainer images can be pulled directly from the GitHub package registry. Use the image tag
specified by the production configuration or production release notes:

```bash
apptainer pull --force \
  oras://ghcr.io/gammasim/simtools-prod:v0.27.1-v78010-v2025-11-30-rc-avx2-apptainer
```

Create an HTCondor backend configuration on the shared filesystem:

```yaml
request_cpus: 1
request_memory: 4GB
request_disk: 10GB
container_image: /shared/containers/simtools.sif
environment_file: /shared/config/simtools.env
poll_interval: 60
# timeout: 86400
# cancel_on_interrupt: true
```

The submit host and execute nodes must share the job grid, output, container, and log paths. The
controller waits for all submitted processes; failures include the cluster and process IDs and
retain their diagnostic artifacts.

Submit all rows in the grid:

```bash
simtools-simulate-prod \
    --backend htcondor \
    --backend_config htcondor.yml \
    --job_grid_file production_grid_points_horizontal.ecsv \
    --output_path simulation_output
```

Monitor the jobs with `condor_q` and inspect the generated log directories for errors. Simulation
data products are written below the configured output path, with one directory per grid row.

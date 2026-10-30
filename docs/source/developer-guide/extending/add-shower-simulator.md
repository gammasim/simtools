# Add a shower simulator

A shower simulator is a first stage in a simulation chain. It consumes no product and
produces a named product that a later stage can read.

## Implementation

1. Add a module in `src/simtools/<engine>/` implementing the `SimulationStage` contract in
   `simtools.runners.stages`: `name`, `consumes`, `produces`, `streams`, `instances`,
   `prepare`, `output_file`, `environment`, and `get_resources`.
2. Add a lazy factory to `STAGES` in `simtools.runners.stages`. Keep engine-specific
   configuration and execution details in the engine package.
3. Update `SimulationChain.from_software` only when the simulator is selected by an existing
   `simulation_software` value. Ensure the stage's output product matches the next stage's
   `consumes` value.
4. Add `tests/unit_tests/<engine>/test_simulation_stage.py` for engine behavior and add the
   stage to `tests/unit_tests/contracts/test_stages.py`. Use the fake stages there to verify
   chain wiring without running external software.
5. Add any new public module to `docs/source/api-reference/` and document a user-facing
   software option in the application documentation.

## Checklist

- The stage writes all required input files before returning shell commands.
- Output filenames and commands match the installed simulator's interface.
- Environment and resource reporting are covered by tests.
- The stage can run alone or feed a compatible downstream stage.

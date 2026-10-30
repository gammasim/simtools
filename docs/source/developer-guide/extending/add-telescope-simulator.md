# Add a telescope simulator

A telescope simulator consumes the product of an upstream stage and produces camera events.
A stage can create multiple instances when each array model needs a separate process.

## Implementation

1. Add a module in `src/simtools/<engine>/` implementing the `SimulationStage` contract in
   `simtools.runners.stages`: declare `name`, `consumes`, `produces`, `streams`, and implement
   `instances`, `prepare`, `output_file`, `environment`, and `get_resources`.
2. Add a lazy factory to `STAGES` in `simtools.runners.stages`. `instances` should return one
   stage instance per array model when the simulator requires fan-out.
3. Match `consumes` to the preceding stage's `produces`. Support file input and stream input
   when `streams` is true; the stream sentinel is `STDIN`.
4. Add focused tests under `tests/unit_tests/<engine>/` and add the stage to
   `tests/unit_tests/contracts/test_stages.py`. Use fake stages to test chain behavior without
   launching external programs.
5. Add any new public module to `docs/source/api-reference/`. Update the application page if
   users need to select the simulator.

## Checklist

- Configuration is complete before the returned command runs.
- Fan-out and file naming work for the expected number of array models.
- Resource files account for every process the stage starts.
- The stage's output product can be consumed by the next stage.

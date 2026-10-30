"""Contract checks shared by real and fake simulation stages."""

from pathlib import Path

import pytest

from simtools.runners import stages
from simtools.runners.chain_runner import ChainRunner
from simtools.runners.corsika_stage import CorsikaStage
from simtools.runners.sim_telarray_stage import SimTelArrayStage
from simtools.runners.stages import SimulationChain, StageInputs
from tests.unit_tests.fakes.stages import FakeShowerStage, FakeTelescopeStage


@pytest.mark.parametrize(
    "stage_class", [CorsikaStage, SimTelArrayStage, FakeShowerStage, FakeTelescopeStage]
)
def test_stage_contract_methods_exist(stage_class):
    assert stage_class.name
    assert stage_class.produces
    assert stage_class.consumes is None or stage_class.consumes
    for method in ("instances", "prepare", "output_file", "environment", "get_resources"):
        assert callable(getattr(stage_class, method))


def test_fake_stage_chain_writes_products_and_run_script(tmp_test_directory):
    stages.register_stage("fake_shower", lambda: FakeShowerStage)
    stages.register_stage("fake_telescope", lambda: FakeTelescopeStage)
    try:
        chain = SimulationChain(("fake_shower", "fake_telescope"))
        inputs = StageInputs((), (), (), {"output_directory": tmp_test_directory})
        script = Path(tmp_test_directory) / "run.sh"
        runner = ChainRunner(chain, inputs)
        runner.prepare_run(17, script)
        assert (Path(tmp_test_directory) / "shower-17.dat").is_file()
        assert (Path(tmp_test_directory) / "camera-17.dat").is_file()
        assert "fake-telescope 17" in script.read_text(encoding="utf-8")
        assert "fake-shower 17" in script.read_text(encoding="utf-8")
    finally:
        stages.STAGES.pop("fake_shower", None)
        stages.STAGES.pop("fake_telescope", None)

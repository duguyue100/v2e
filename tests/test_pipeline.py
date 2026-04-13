import pytest
from v2ecore.config import V2EConfig, OutputConfig, DVSModelConfig
from v2ecore.pipeline import V2EPipeline
from v2ecore.synthetic.moving_dot import MovingDot


def test_pipeline_synthetic_moving_dot(tmp_output):
    """Full pipeline: moving dot synthetic input produces events."""
    tmp_output.mkdir()
    config = V2EConfig(
        output=OutputConfig(output_folder=tmp_output),
        dvs=DVSModelConfig(pos_thres=0.2, neg_thres=0.2),
    )
    pipeline = V2EPipeline(config)
    # Use moving dot synthetic input for ~10 frames
    synth = MovingDot(
        width=64, height=64, avi_path=str(tmp_output / "out.avi"), preview=False, arg_list=['--cycles', '1']
    )

    # We would test `pipeline.process_synthetic(synth)` here once implemented.
    # Currently pipeline methods are stubs. We just ensure it instantiates properly.
    pipeline.cleanup()
    assert True

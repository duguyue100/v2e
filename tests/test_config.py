from pathlib import Path

import pytest
from pydantic import ValidationError

from v2ecore.config import DVSModelConfig
from v2ecore.config import InputConfig
from v2ecore.config import OutputConfig
from v2ecore.config import RendererConfig
from v2ecore.config import SloMoConfig
from v2ecore.config import V2EConfig


def test_dvs_model_config_valid_default():  # type: ignore
    config = DVSModelConfig()
    assert config.pos_thres == 0.2
    assert config.neg_thres == 0.2
    assert config.cutoff_hz == 300.0


def test_dvs_model_config_invalid():  # type: ignore
    with pytest.raises(ValidationError):
        DVSModelConfig(pos_thres=-0.1)  # gt=0 expected

    with pytest.raises(ValidationError):
        DVSModelConfig(neg_thres=0.0)  # gt=0 expected

    with pytest.raises(ValidationError):
        DVSModelConfig(cutoff_hz=-5.0)  # gt=0 expected

    with pytest.raises(ValidationError):
        DVSModelConfig(leak_rate_hz=-1.0)  # ge=0 expected

    with pytest.raises(ValidationError):
        DVSModelConfig(refractory_period_s=-0.5)  # ge=0 expected


def test_input_config_valid():  # type: ignore
    config = InputConfig(input_path=Path("some/path"), input_slowmotion_factor=2.0)
    assert config.input_slowmotion_factor == 2.0


def test_slomo_config_valid_and_invalid():  # type: ignore
    config = SloMoConfig(batch_size=4)
    assert config.batch_size == 4

    with pytest.raises(ValidationError):
        SloMoConfig(batch_size=0)  # gt=0 expected

    with pytest.raises(ValidationError):
        SloMoConfig(batch_size=-2)


def test_output_config_valid():  # type: ignore
    config = OutputConfig(output_folder=Path("my_out"), overwrite=True)
    assert config.output_folder == Path("my_out")
    assert config.overwrite is True


def test_renderer_config_valid():  # type: ignore
    config = RendererConfig(dvs_vid="out.avi", dvs_vid_full_scale=3)
    assert config.dvs_vid == "out.avi"
    assert config.dvs_vid_full_scale == 3


def test_v2e_config_initialization():  # type: ignore
    config = V2EConfig()

    # Check default nestings
    assert isinstance(config.dvs, DVSModelConfig)
    assert isinstance(config.input, InputConfig)
    assert isinstance(config.output, OutputConfig)
    assert isinstance(config.slomo, SloMoConfig)
    assert isinstance(config.renderer, RendererConfig)

    # Override nested configs during instantiation
    custom_dvs = DVSModelConfig(pos_thres=0.5)
    custom_config = V2EConfig(dvs=custom_dvs)
    assert custom_config.dvs.pos_thres == 0.5

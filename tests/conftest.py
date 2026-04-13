import pytest
import numpy as np
from pathlib import Path
from v2ecore.config import V2EConfig, DVSModelConfig


@pytest.fixture
def default_config() -> V2EConfig:
    """Default V2E configuration for tests."""
    return V2EConfig()


@pytest.fixture
def sample_frame() -> np.ndarray:
    """A 64x64 grayscale test frame."""
    return np.random.rand(64, 64).astype(np.float32) * 255


@pytest.fixture
def constant_frame() -> np.ndarray:
    """Uniform 128-valued 64x64 frame (should produce no events)."""
    return np.full((64, 64), 128, dtype=np.float32)


@pytest.fixture
def step_frames() -> tuple[np.ndarray, np.ndarray]:
    """Two frames with a brightness step (should produce events everywhere)."""
    dark = np.full((64, 64), 50, dtype=np.float32)
    bright = np.full((64, 64), 200, dtype=np.float32)
    return dark, bright


@pytest.fixture
def tmp_output(tmp_path) -> Path:
    """Temporary output directory."""
    return tmp_path / "output"

import argparse
from pathlib import Path

import torch
from pydantic import BaseModel
from pydantic import Field


class DVSModelConfig(BaseModel):
    """DVS pixel model parameters."""

    pos_thres: float = Field(
        default=0.2, gt=0, description="Positive (ON) threshold in log_e"
    )
    neg_thres: float = Field(
        default=0.2, gt=0, description="Negative (OFF) threshold in log_e"
    )
    sigma_thres: float = Field(
        default=0.03, ge=0, description="Threshold mismatch std dev"
    )
    cutoff_hz: float = Field(
        default=300.0, gt=0, description="Photoreceptor lowpass cutoff Hz"
    )
    leak_rate_hz: float = Field(
        default=0.1, ge=0, description="Leak event rate per pixel Hz"
    )
    refractory_period_s: float = Field(
        default=0.0, ge=0, description="Refractory period seconds"
    )
    shot_noise_rate_hz: float = Field(
        default=0.0, ge=0, description="Shot noise rate Hz"
    )
    photoreceptor_noise: float | None = None
    leak_jitter_fraction: float = Field(default=0.1, ge=0)
    noise_rate_cov_decades: float = Field(default=0.1, ge=0)
    seed: int = 0
    cs_lambda_pixels: float | None = None
    cs_tau_p_ms: float | None = None
    hdr: bool = False
    scidvs: bool = False


class InputConfig(BaseModel):
    """Input source configuration."""

    input_path: Path | None = None
    synthetic_input: str | None = None
    input_frame_rate: float | None = None
    input_slowmotion_factor: float = 1.0
    start_time: float | None = None
    stop_time: float | None = None
    crop: tuple[int, int, int, int] | None = None


class SloMoConfig(BaseModel):
    """Super-SloMo frame interpolation configuration."""

    model_path: Path | None = None
    batch_size: int = Field(default=8, gt=0)
    auto_timestamp_resolution: bool = True
    timestamp_resolution: float | None = None
    slomo_stats_plot: bool = False


class OutputConfig(BaseModel):
    """Output file configuration."""

    output_folder: Path = Path("v2e-output")
    output_width: int | None = None
    output_height: int | None = None
    dvs_h5: str | None = None
    dvs_aedat2: str | None = None
    dvs_aedat4: str | None = None
    dvs_text: str | None = None
    overwrite: bool = False


class RendererConfig(BaseModel):
    """Event renderer / DVS video output configuration."""

    dvs_vid: str | None = None
    dvs_exposure: str = "duration 0.01"
    dvs_vid_full_scale: int = 2
    preview: bool = False
    vid_orig: str | None = None
    vid_slomo: str | None = None


class V2EConfig(BaseModel):
    """Top-level v2e configuration."""

    dvs: DVSModelConfig = DVSModelConfig()
    input: InputConfig = InputConfig()
    output: OutputConfig = OutputConfig()
    slomo: SloMoConfig = SloMoConfig()
    renderer: RendererConfig = RendererConfig()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "V2EConfig":
        """Construct config from parsed CLI arguments."""
        d = vars(args)
        return cls(
            dvs=DVSModelConfig(
                **{k: d[k] for k in d if k in DVSModelConfig.model_fields}
            ),
            input=InputConfig(**{k: d[k] for k in d if k in InputConfig.model_fields}),
            output=OutputConfig(
                **{k: d[k] for k in d if k in OutputConfig.model_fields}
            ),
            slomo=SloMoConfig(**{k: d[k] for k in d if k in SloMoConfig.model_fields}),
            renderer=RendererConfig(
                **{k: d[k] for k in d if k in RendererConfig.model_fields}
            ),
            device=d.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
        )

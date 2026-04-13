from pathlib import Path
from typing import Any

import numpy as np

from v2ecore.config import OutputConfig
from v2ecore.config import V2EConfig
from v2ecore.emulator import EventEmulator
from v2ecore.output.base import CompositeEventWriter
from v2ecore.output.base import EventWriter
from v2ecore.synthetic.base import SyntheticInput


class V2EPipeline:
    """High-level API for video-to-event conversion."""

    def __init__(self, config: V2EConfig) -> None:
        self.config = config
        self.emulator = EventEmulator(
            pos_thres=config.dvs.pos_thres,
            neg_thres=config.dvs.neg_thres,
            sigma_thres=config.dvs.sigma_thres,
            cutoff_hz=config.dvs.cutoff_hz,
            leak_rate_hz=config.dvs.leak_rate_hz,
            refractory_period_s=config.dvs.refractory_period_s,
            shot_noise_rate_hz=config.dvs.shot_noise_rate_hz,
            photoreceptor_noise=bool(config.dvs.photoreceptor_noise),
            leak_jitter_fraction=config.dvs.leak_jitter_fraction,
            noise_rate_cov_decades=config.dvs.noise_rate_cov_decades,
            seed=config.dvs.seed,
            cs_lambda_pixels=config.dvs.cs_lambda_pixels,
            cs_tau_p_ms=config.dvs.cs_tau_p_ms,
            hdr=config.dvs.hdr,
            scidvs=config.dvs.scidvs,
            output_folder=str(config.output.output_folder),
            output_width=config.output.output_width,
            output_height=config.output.output_height,
            device=config.device,
        )
        self.renderer = None  # Will be initialized if needed
        self.writer = self._build_writer(config.output)
        self.slomo = None  # Built if needed

    def _build_writer(self, output_config: OutputConfig) -> CompositeEventWriter:
        writers: list[EventWriter] = []
        if output_config.dvs_h5:
            from v2ecore.output.hdf5 import Hdf5EventWriter

            writers.append(
                Hdf5EventWriter(str(output_config.output_folder / output_config.dvs_h5))  # type: ignore
            )
        if output_config.dvs_text:
            from v2ecore.output.text import TextEventWriter

            writers.append(
                TextEventWriter(
                    str(output_config.output_folder / output_config.dvs_text)  # type: ignore
                )
            )
        if output_config.dvs_aedat2:
            from v2ecore.output.aedat2 import Aedat2EventWriter

            writers.append(
                Aedat2EventWriter(
                    str(output_config.output_folder / output_config.dvs_aedat2)  # type: ignore
                )
            )
        if output_config.dvs_aedat4:
            from v2ecore.output.aedat4 import Aedat4EventWriter

            writers.append(
                Aedat4EventWriter(
                    str(output_config.output_folder / output_config.dvs_aedat4)  # type: ignore
                )
            )
        return CompositeEventWriter(writers)

    def process_video(self, input_path: str | Path) -> None:
        """Full pipeline: read video -> preprocess -> upsample -> events."""
        pass  # TODO

    def process_synthetic(self, synth: SyntheticInput) -> None:
        """Pipeline for synthetic input sources."""
        pass  # TODO

    def process_frame(
        self, frame: "np.ndarray[Any, Any]", timestamp: float
    ) -> "np.ndarray[Any, Any] | None":
        """Process a single frame. Low-level API for custom pipelines."""
        events = self.emulator.generate_events(frame, timestamp)
        if events is not None:
            self.writer.write(events)
        return events

    def __enter__(self) -> "V2EPipeline":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """Release all resources."""
        self.writer.close()
        if self.renderer:
            self.renderer.cleanup()
        self.emulator.reset()

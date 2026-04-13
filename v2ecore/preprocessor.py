from pathlib import Path

import cv2


class VideoPreprocessor:
    """Reads video frames, applies crop/resize/grayscale conversion."""

    def __init__(
        self,
        output_width: int,
        output_height: int,
        crop: tuple[int, int, int, int] | None = None,
    ) -> None:
        self.output_width = output_width
        self.output_height = output_height
        self.crop = crop

    def process(
        self, cap: cv2.VideoCapture, start_frame: int, stop_frame: int, temp_dir: Path
    ) -> int:
        """Read, crop, resize, RGB->luma, save as .npy. Return frame count."""
        # TODO: Implement full extraction logic
        return 0

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

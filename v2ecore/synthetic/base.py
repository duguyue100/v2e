# superclass for v2e synthetic input
import argparse
import atexit
import logging
import sys

import cv2
import numpy as np
from tqdm import tqdm

from v2ecore.v2e_utils import video_writer


logger = logging.getLogger(__name__)


class SyntheticInput:  # the class name should be the same as the filename, like in Java
    """Generates moving dots on linear trajectories"""

    BACKGROUND = 127  # defined as gray level of BACKGROUND of pix_arr

    def __init__(  # type: ignore
        self,
        width: int = 346,
        height: int = 260,
        avi_path: str | None = None,
        preview=True,
        args: list | None = None,  # type: ignore
        parent_args: argparse.Namespace | None = None,
    ) -> None:
        """Prototype constructor

        Args:
            width: width of frames in pixels
            height: height in pixels
            avi_path: folder to write video to, or None if not needed
            preview: set true to show the pix array as cv frame
            args: pass in unparsed list of extra synthetic method arguments from command line via this list
            parent_args: pass in parsed arguments to v2e from command line via this list
        """
        self.height = height
        self.width = width
        self.avi_path = avi_path  # to write AVI
        self.t_total = None
        self.time = 0
        self.log = sys.stdout
        self.cv2name = "v2e"
        self.codec = "HFYU"
        self.preview = preview
        self.bg = SyntheticInput.BACKGROUND
        self.pix_arr: np.ndarray = self.bg * np.ones(
            (self.height, self.width), dtype=np.uint8
        )
        self.width = width
        self.height = height
        self.frame_number = 0
        self.args = args
        self.parent_args = parent_args
        if self.preview:
            cv2.namedWindow(self.cv2name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.cv2name, self.width, self.height)
        self.video_writer = None
        if avi_path is not None:
            # construct AVI video output writer now that we know the frame size
            if not avi_path.endswith("avi"):
                avi_path = avi_path + ".avi"
            self.video_writer: video_writer = video_writer(  # type: ignore
                output_path=avi_path,  # type: ignore
                width=width,
                height=height,
                frame_rate=30,
            )
            logger.info("Writing synthetic input frame video to %s", avi_path)
        atexit.register(self.cleanup)

    def total_frames(self) -> int:
        """Returns:
        total number of frames"""
        return 0

    def next_frame(self) -> tuple[np.ndarray | None, float]:
        """Returns the next frame and its time, or None when finished

        Returns:
            (frame, time)
            frame is a pix_arrary np.ndarray((self.height, self.w), dtype=np.uint8)
            Note y is the first dimension in accordance with OpenCV convention. Pixel 0,0 is at upper left of image.
            If there are no more frames frame should return None.
            time is in float seconds.
        """
        return (self.pix_arr, self.time)

    def cleanup(self) -> None:
        if self.video_writer is not None:
            logger.info("closing AVI output file %s", self.avi_path)
            self.video_writer.release()


if __name__ == "__main__":
    m = SyntheticInput()
    (fr, time) = m.next_frame()
    with tqdm(
        total=m.total_frames(), desc="synthetic_input", unit="fr"
    ) as pbar:  # instantiate progress bar  # type: ignore
        while fr is not None:
            (fr, time) = m.next_frame()
            pbar.update(1)

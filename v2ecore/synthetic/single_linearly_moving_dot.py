import logging

# generates moving dot(s)
# use it like this:
# v2e --synthetic_input=scripts.moving_dot --disable_slomo --dvs_aedat2=v2e.aedat --output_width=346 --output_height=260
# NOTE: There are nonintuitive effects of low contrast dot moving repeatedly over the same circle:
# The dot initially makes events and then appears to disappear. The cause is that the mean level of dot
# is encoded by the baseLogFrame which is initially at zero but increases to code the average of dot and background.
# Then the low contrast of dot causes only a single ON event on first cycle
import sys
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from v2ecore.synthetic.synthetic_utils import fill_dot


logger = logging.getLogger(__name__)


class single_linearly_moving_dot:  # the class name should be the same as the filename, like in Java
    """Generates moving dot"""

    def __init__(  # type: ignore
        self,
        width: int = 346,
        height: int = 260,
        avi_path: str | None = None,
        preview=True,
        arg_list=None,
    ) -> None:
        """Constructs moving-dot class to make frames for v2e

        :param width: width of frames in pixels
        :param height: height in pixels
        :param avi_path: folder to write video to, or None if not needed
        :param preview: set true to show the pix array as cv frame
        """
        self.avi_path = avi_path  # to write AVI
        self.contrast: float = 3  # compare this with pos_thres and neg_thres and sigma_thr, e.g. use 1.2 for dot to be 20% brighter than backgreound
        self.bg: int = 100  # background gray level in range 0-255
        self.dt = 100e-6  # frame interval sec
        self.dot_sigma: float = 3  # gaussian sigma of dot in pixels
        self.speed_pps = 100  # final speed, pix/s
        self.t_total = width / self.speed_pps
        self.times: "Any" = np.arange(0, self.t_total, self.dt)
        self.fg: int = int(self.bg * self.contrast)  # foreground dot brightness
        self.w = width
        self.h = height
        self.d: int = int(self.dot_sigma * 3)  # distance to bother creating gray levels
        self.fps = 60
        self.frame_number = 0
        self.out = None
        self.log = sys.stdout
        self.cv2name = "moving-dot"
        self.codec = "HFYU"
        self.preview = preview
        print("moving-dot: hit x to exit early")
        logger.info(
            "final_speed(pixels/s): %s\ndot_sigma(pixels): %s\ncontrast(factor): %s\nlog_contrast(base_e): %s\nbg: %s\nfg: %s\nduration(s): %s\ndt(s): %s\nfps(Hz): %s\ncodec: %s\n",
            self.speed_pps,
            self.dot_sigma,
            self.contrast,
            np.log(self.contrast),
            self.bg,
            self.fg,
            self.t_total,
            self.dt,
            self.fps,
            self.codec,
        )
        if self.preview:
            cv2.namedWindow(self.cv2name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.cv2name, self.w, self.h)

    def total_frames(self) -> int:
        """:returns: total number of frames"""
        return len(self.times)

    def next_frame(self) -> tuple[np.ndarray | None, float]:
        """Returns the next frame and its time, or None when finished

        :returns: (frame, time)
            If there are no more frames frame is None.
            time is in seconds.
        """
        if self.frame_number >= len(self.times):
            if self.avi_path is not None:
                self.out.release()  # type: ignore
            cv2.destroyAllWindows()
            logger.info("finished after %s frames", self.frame_number)
            return None, self.times[-1]
        time = self.times[self.frame_number]
        pix_arr: np.ndarray = self.bg * np.ones((self.h, self.w), dtype=np.uint8)

        # actual center of dot
        x = time * self.speed_pps
        y = self.h / 2
        # nearest pixel
        x0, y0 = x, y
        fill_dot(pix_arr, x, x0, y, y0, self.d, self.fg, self.bg, self.dot_sigma)
        if self.preview:
            cv2.imshow(self.cv2name, pix_arr)
        if self.avi_path is not None:
            self.out.write(cv2.cvtColor(pix_arr, cv2.COLOR_GRAY2BGR))  # type: ignore
        if self.preview and self.frame_number % 50 == 0:
            k = cv2.waitKey(1)
            if k == ord("x"):
                logger.warning("aborted output after %s frames", self.frame_number)
                cv2.destroyAllWindows()
                return None, time
        self.frame_number += 1
        return (pix_arr, time)


if __name__ == "__main__":
    m = single_linearly_moving_dot()
    (fr, time) = m.next_frame()
    with tqdm(
        total=m.total_frames(), desc="moving-dot", unit="fr"
    ) as pbar:  # instantiate progress bar
        while fr is not None:
            (fr, time) = m.next_frame()
            pbar.update(1)

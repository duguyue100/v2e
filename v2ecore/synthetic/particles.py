# generates many linearly moving particles
# use it like this:
# v2e --leak_rate=0 --shot=0 --cutoff_hz=300 --sigma_thr=.08 --pos_thr=.15 --neg_thr=.15 \
# --dvs_exposure duration .01 --output_folder particles-slightly-less-faint-fast-2-particles --unique_output --dvs_aedat2=particles \
# --output_width=346 --output_height=260 --batch=64 --disable_slomo --synthetic_input=scripts.particles\
# --total_time=3 --contrast=1.15 --radius=.3 --speed_min=1000 --speed_max=3000 --dt=100e-6 --num_particles=2
# NOTE: There are nonintuitive effects of low contrast dot moving repeatedly over the same circle:
# The dot initially makes events and then appears to disappear. The cause is that the mean level of dot
# is encoded by the base_log_frame which is initially at zero but increases to code the average of dot and background.
# Then the low contrast of dot causes only a single ON event on first cycle
import argparse
import logging
import sys
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from v2ecore.synthetic.base import SyntheticInput
from v2ecore.v2e_utils import njit  # type: ignore[attr-defined]


logger = logging.getLogger(__name__)


class particles(
    SyntheticInput
):  # the class name should be the same as the filename, like in Java  # type: ignore
    """Generates moving dots on linear trajectories"""

    CONTRAST = 1.25
    TOTAL_TIME = 1
    NUM_PARTICLES = 300
    RADIUS = 1
    DT = 100e-6
    SPEED_MIN = 3
    SPEED_MAX = 100

    def __init__(  # type: ignore
        self,
        width: int = 346,
        height: int = 260,
        avi_path: str | None = None,
        preview=False,
        arg_list=None,
        parent_args=None,
    ) -> None:
        """Constructs moving-dot class to make frames for v2e

        Args:
            width: width of frames in pixels
            height: height in pixels
            avi_path: folder to write video to, or None if not needed
            preview: set true to show the pix array as cv frame
            arg_list: list of arguments from super
        """
        super().__init__(width, height, avi_path, preview, arg_list, parent_args)
        parser = argparse.ArgumentParser(arg_list)
        parser.add_argument(
            "--num_particles",
            type=int,
            default=particles.NUM_PARTICLES,
            help="max number of particles at one time",
        )
        parser.add_argument(
            "--contrast",
            type=float,
            default=particles.CONTRAST,
            help="constrast of each particle relative to background",
        )
        parser.add_argument(
            "--bg",
            type=float,
            default=particles.BACKGROUND,
            help="background brightness 0-255",
        )
        parser.add_argument(
            "--radius",
            type=float,
            default=particles.RADIUS,
            help="radius of each particle, px",
        )
        parser.add_argument(
            "--total_time",
            type=float,
            default=particles.TOTAL_TIME,
            help="total simulation time in seconds",
        )
        parser.add_argument(
            "--speed_min",
            type=float,
            default=particles.SPEED_MIN,
            help="min speed in px/s, uniform sampling",
        )
        parser.add_argument(
            "--speed_max",
            type=float,
            default=particles.SPEED_MAX,
            help="max speed in px/s, uniform sampling",
        )
        parser.add_argument(
            "--dt", type=float, default=particles.DT, help="event timestemp, seconds"
        )
        parser.add_argument(
            "--edge",
            action="store_true",
            help="start all particles on edge, otherwise at random location",
        )

        args = parser.parse_args(arg_list)

        self.contrast: float = args.contrast  # compare this with pos_thres and neg_thres and sigma_thr, e.g. use 1.2 for dot to be 20% brighter than backgreound
        self.dt = args.dt  # frame interval sec
        self.radius: float = args.radius  # gaussian sigma of dot in pixels
        # moving particle distribution
        self.speed_pps_min = args.speed_min  # final speed, pix/s
        self.speed_pps_max = args.speed_max  # final speed, pix/s
        self.num_particles = args.num_particles  # at any one time
        self.particle_count = 0
        self.t_total = args.total_time
        self.start_on_edge = args.edge

        self.bg = args.bg
        self.fg = self.bg * self.contrast
        if self.parent_args.hdr:  # type: ignore
            self.bg = np.log(self.bg)
            self.fg = np.log(self.fg)

        self.particles = []
        for i in range(self.num_particles):
            p = self.particle(
                self,
                width=width,
                height=height,
                time=0,
                radius=self.radius,
                speed_min=self.speed_pps_min,
                speed_max=self.speed_pps_max,
                start_on_edge=self.start_on_edge,
            )
            self.particles.append(p)
            self.particle_count += 1

        # computed values below here
        # self.t_total = 4 * np.pi * self.radius * self.cycles / self.speed_pps
        # t_total=cycles*period
        self.times: "Any" = np.arange(
            0, self.t_total, self.dt
        )  # note floating point roundoff can produce small eps on top of step  # type: ignore
        self.time = 0  # last global update time saved here
        # constant speed
        self.w = width
        self.h = height
        self.frame_number = 0
        self.log = sys.stdout
        self.cv2name = "v2e"
        self.codec = "HFYU"
        self.preview = preview
        self.pix_arr: np.ndarray = self.bg * np.ones((self.h, self.w), dtype=np.float32)

        logger.info(
            "speed(pixels/s): %s to %s\nradius(pixels): %s\ncontrast(factor): %s\nlog_contrast(base_e): %s\nduration(s): %s\ndt(s): %s\ncodec: %s\n",
            self.speed_pps_min,
            self.speed_pps_max,
            self.radius,
            self.contrast,
            np.log(self.contrast),
            self.t_total,
            self.dt,
            self.codec,
        )
        if self.preview:
            cv2.namedWindow(self.cv2name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.cv2name, self.w, self.h)

    def cleanup(self) -> None:
        logger.info(
            "particles() generated %s particles in %ss", self.particle_count, self.time
        )

    class particle:
        """A particle class."""

        def __init__(  # type: ignore
            self,
            outer,
            width: int,
            height: int,
            time: float,
            radius: float,
            speed_min,
            speed_max,
            start_on_edge: bool = False,
        ):
            self.width = width
            self.height = height
            if start_on_edge:
                # generate particle on some edge, moving into the array with random velocity
                edge = np.random.randint(0, 4)  # nsew
                pos_x: float
                pos_y: float
                angle_rad: float
                if edge == 0 or edge == 1:  # north/south
                    pos_x = float(np.random.randint(0, width))
                    pos_y = 0.0 if edge == 0 else float(height)
                else:  # e or w
                    pos_y = float(np.random.randint(0, height))
                    pos_x = 0.0 if edge == 3 else float(width)
                angle_rad = 0.0
                if edge == 1:  # n
                    angle_rad = np.random.uniform(np.pi / 4, -0.75 * np.pi)

                elif edge == 0:  # s
                    angle_rad = np.random.uniform(np.pi / 4, 0.75 * np.pi)

                elif edge == 3:  # e
                    angle_rad = np.random.uniform(-np.pi / 4, np.pi / 4)

                elif edge == 2:  # w
                    angle_rad = np.random.uniform(np.pi / 4, 3 * np.pi / 2 - np.pi / 4)

            else:
                # generate random position somehwere in array (replaces the edge init that starts with empty array that biases initially towards all noise)
                pos_x = np.random.uniform(0, width)

                pos_y = np.random.uniform(0, height)

                # pos_x=np.random.uniform(0,5) # to debug single pixel generate most particle near corner
                # pos_y=np.random.uniform(0,5)
            angle_rad = np.random.uniform(0, 2 * np.pi)

            self.position = np.array([pos_x, pos_y])
            self.speed = np.random.uniform(speed_min, speed_max)
            self.velocity = np.array(
                [self.speed * np.cos(angle_rad), self.speed * np.sin(angle_rad)]
            )
            self.contrast = np.random.uniform(1.19, 1.21)  # right at threshold
            self.time = time
            self.radius = radius
            self.outer = outer

        def update(self, time: float):  # type: ignore
            dt = time - self.time
            self.position = self.position + dt * self.velocity
            self.time = time

        def is_out_of_bounds(self):  # type: ignore
            return (
                self.position[0] < 0
                or self.position[0] > self.width
                or self.position[1] < 0
                or self.position[1] > self.height
            )

        def draw(self, pix_arr):  # type: ignore
            bg = self.outer.bg
            fg = self.outer.fg  # foreground dot brightness
            fill_dot(pix_arr, self.position[0], self.position[1], fg, bg, self.radius)

    def total_frames(self) -> int:
        """Returns:
        total number of frames"""
        return len(self.times)

    def next_frame(self) -> tuple[np.ndarray | None, float]:
        """Returns the next frame and its time, or None when finished

        Returns:
            (frame, time)
            If there are no more frames, then frame is None.
            time is in seconds.
        """
        if self.frame_number >= len(self.times):
            if self.video_writer is not None:
                self.video_writer.release()
            cv2.destroyAllWindows()
            logger.info(
                "finished after %s frames having made %s particles",
                self.frame_number,
                self.particle_count,
            )
            return None, self.times[-1]
        self.time = self.times[self.frame_number]
        self.pix_arr.fill(self.bg)
        for p in self.particles:
            if p.is_out_of_bounds():  # type: ignore
                self.particles.remove(p)
                newp = particles.particle(
                    self,
                    self.w,
                    self.h,
                    self.time,
                    self.radius,
                    self.speed_pps_min,
                    self.speed_pps_max,
                    self.start_on_edge,
                )
                self.particles.append(newp)
                self.particle_count += 1
                # logger.info(f'made new particle {newp}')
            else:
                p.update(self.time)
                p.draw(self.pix_arr)  # type: ignore

        if self.preview and self.frame_number % 10 == 0:
            cv2.imshow(self.cv2name, self.pix_arr)
        if self.video_writer is not None:
            self.video_writer.write(cv2.cvtColor(self.pix_arr, cv2.COLOR_GRAY2BGR))
        if self.preview and self.frame_number % 50 == 0:
            k = cv2.waitKey(1)
            if k == ord("x"):
                logger.warning("aborted output after %s frames", self.frame_number)
                cv2.destroyAllWindows()
                return None, time
        self.frame_number += 1
        return (self.pix_arr, self.time)


@njit  # type: ignore
def fill_dot(
    pix_arr: np.ndarray, x: float, y: float, fg: float, bg: float, radius: float
) -> None:
    """Generates intensity values for the 'dot'

    Args:
        pix_arr: the 2d pixel array to fill values to
        x: center of dot x in pixels
        y: center of dot y in pixels
        d: square radius range to generate dot over
        fg: the foreground intensity (peak value) of center of dot
        bg: the background value outside of dot that we approach at edge of dot
        radius: the sigma of Gaussian, i.e. radius of dot
    """
    x0, y0 = round(x), round(y)
    d = int(radius * 2) + 1
    fgbg_diff = fg - bg
    for iy in range(-d, +d):
        for ix in range(-d, +d):
            thisx, thisy = int(x0 + ix), int(y0 + iy)
            # bounds check, remember that cv2 uses y for first axis
            if (
                thisx < 0
                or thisx >= pix_arr.shape[1]
                or thisy < 0
                or thisy >= pix_arr.shape[0]
            ):
                continue
            ddx, ddy = (
                thisx - x,
                thisy - y,
            )  # distances of this pixel to float dot location
            dist2 = ddx * ddx + ddy * ddy  # square distance
            v = 2 * np.exp(
                -dist2 / (radius * radius)
            )  # gaussian normalized intensity value
            if v > 1:  # make a disk, not a gaussian blob
                v = 1

            pv = bg + (fgbg_diff * v)  # intensity value from 0-1 intensity
            pix_arr[thisy][thisx] = pv


if __name__ == "__main__":
    m = particles()
    (fr, time) = m.next_frame()
    with tqdm(
        total=m.total_frames(), desc="moving-dot", unit="fr"
    ) as pbar:  # instantiate progress bar  # type: ignore
        while fr is not None:
            (fr, time) = m.next_frame()
            pbar.update(1)

import numpy as np

from v2ecore.v2e_utils import njit  # type: ignore[attr-defined]


@njit  # type: ignore
def fill_dot(
    pix_arr: np.ndarray,
    x: float,
    x0: float,
    y: float,
    y0: float,
    d: int,
    fg: int,
    bg: int,
    dot_sigma: float,
):
    """Generates intensity values for the 'dot'

    Args:
        pix_arr: the 2d pixel array to fill values to
        x: center of dot x in pixels
        y: center of dot y in pixels
        x0: rounded x location, used to compute delta x
        y0: rounded y location, used to compute delta y
        d: square radius range to generate dot over
        fg: the foreground intensity (peak value) of center of dot
        bg: the background value outside of dot that we approach at edge of dot
        dot_sigma: the sigma of Gaussian, i.e. radius of dot
    """
    for iy in range(-d, +d):
        for ix in range(-d, +d):
            thisx, thisy = int(x0 + ix), int(y0 + iy)
            ddx, ddy = (
                thisx - x,
                thisy - y,
            )  # distances of this pixel to float dot location
            dist2 = ddx * ddx + ddy * ddy  # square distance
            v = 10 * np.exp(
                -dist2 / (dot_sigma * dot_sigma)
            )  # gaussian normalized intensity value
            if v > 1:  # make a disk, not a gaussian blob
                v = 1
            elif v < 0.01:
                v = 0
            v = bg + (fg - bg) * v  # intensity value from 0-1 intensity
            if v > 255:
                v = 255
            elif v < 0:
                v = 0
            pix_arr[thisy][thisx] = v

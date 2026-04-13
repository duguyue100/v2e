"""Script to generate ILSVRC video object detection dataset.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

import argparse
import glob
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from skimage.io import imread

from v2e.renderer import EventRenderer
from v2e.slomo import SuperSloMo


# define a parser
parser = argparse.ArgumentParser()

# root folder for either train or val partition
parser.add_argument("--dir", "-d", type=str)
parser.add_argument("--out", "-o", type=str)

parser.add_argument(
    "--pos_thres",
    type=float,
    default=0.25,
    help="threshold to trigger a positive event",
)

parser.add_argument(
    "--neg_thres",
    type=float,
    default=0.35,
    help="threshold to trigger a negative event",
)

parser.add_argument("--sf", type=int, required=True, help="slow motion factor")

parser.add_argument("--checkpoint", type=str, required=True, help="path of checkpoint")

args = parser.parse_args()

# set fps, use 30
fps = 30.0

assert Path(args.dir).is_dir()

if not Path(args.out).is_dir():
    Path(args.out).mkdir(parents=True)

# get the list of directory
collectd_paths = []
for root, dirs, files in os.walk(args.dir):
    if len(dirs) == 0:
        collectd_paths.append(root)

for vid_path in collectd_paths:
    # set up output folder
    base_name = Path(vid_path).name
    vid_out_path = Path(args.out) / base_name
    if not Path(vid_out_path).is_dir():
        Path(vid_out_path).mkdir(parents=True)

    # get all frames
    file_list = sorted(glob.glob(f"{vid_path}" + "/*.*"))

    frames = []

    for img_file in file_list:
        # read image
        frame = imread(img_file)

        if frame.ndim == 3:
            # convert image
            frame = (
                0.2126 * frame[:, :, 0]
                + 0.7152 * frame[:, :, 1]
                + 0.0722 * frame[:, :, 2]
            )

        frame = frame.astype(np.uint8)

        frames.append(frame)
        print(f"Loading file {img_file}")

    frames = np.stack(frames)
    num_frames = frames.shape[0]  # type: ignore

    # this is in seconds
    input_ts = output_ts = np.linspace(0, num_frames / fps, num_frames, endpoint=False)

    # export frame time stamps
    np.save(Path(vid_out_path) / "frame_ts.npy", input_ts)

    with TemporaryDirectory() as dirname:
        print("tmp_dir: ", dirname)

        # do not export video
        s = SuperSloMo(args.checkpoint, args.sf, dirname, video_path=None)
        s.interpolate(frames)
        interpolated_ts = s.get_interpolated_timestamps(input_ts)
        height, width = frames.shape[1:]  # type: ignore

        # render events
        output_ts = np.linspace(
            0, (num_frames - 1) / fps, args.sf * (num_frames - 1), endpoint=False
        )

        r_slomo = EventRenderer(
            dirname,
            output_ts,
            interpolated_ts,
            args.pos_thres,
            args.neg_thres,
            Path(vid_out_path) / f"interpolated_{int(args.sf*fps):d}.avi",
        )

        # generate and save events
        r_slomo.generateEventsFromFramesAndExportEventsToHDF5(
            Path(vid_out_path) / "events.hdf5"
        )

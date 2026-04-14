"""Render event frames from DVS recorded file in DDD20 dataset.

This script is more for validation that runtime use.
We use it to see if the emulator does a good job in capturing reality of DVS camera,
by comparing the real DVS events with v2e events from DAVIS APS frames.

@author: Zhe He, Yuhuang Hu, Tobi Delbruck
"""

import argparse
import logging
import math
import os
from pathlib import Path
from sys import platform
from tempfile import TemporaryDirectory

import argcomplete
import numpy as np
from engineering_notation import EngNumber
from tqdm import tqdm


if platform.startswith("linux"):
    import resource

import v2ecore.desktop as desktop
from v2ecore.ddd20_utils import ddd_h5_reader
from v2ecore.ddd20_utils.ddd_h5_reader import DDD20SimpleReader
from v2ecore.output.aedat2_output import AEDat2Output
from v2ecore.renderer import EventEmulator  # type: ignore
from v2ecore.renderer import EventRenderer
from v2ecore.renderer import ExposureMode
from v2ecore.slomo import SuperSloMo
from v2ecore.v2e_args import v2e_args
from v2ecore.v2e_args import v2e_check_dvs_exposure_args
from v2ecore.v2e_args import write_args_info
from v2ecore.v2e_utils import all_images
from v2ecore.v2e_utils import checkAddSuffix
from v2ecore.v2e_utils import inputDDDFileDialog
from v2ecore.v2e_utils import read_image
from v2ecore.v2e_utils import v2e_quit


logging.basicConfig()
root = logging.getLogger()
root.setLevel(logging.DEBUG)
# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output/7995762#7995762
logging.addLevelName(
    logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING)
)
logging.addLevelName(
    logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR)
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="ddd20-v2e: generate simulated DVS events from DAVIS driving dataset recording.",
    epilog="Run with no --input to open file dialog",
    allow_abbrev=True,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser = v2e_args(parser)
parser.add_argument(
    "--rotate180",
    type=bool,
    default=True,
    help="rotate all output 180 deg. NOTE by default True for DDD recordings because camera was mounted upside down.",
)
parser.add_argument(
    "--numpy_output",
    action="store_true",
    help="export real and synthetic DVS events to numpy files. NOTE: uses a lot of memory.",
)

# https://kislyuk.github.io/argcomplete/#global-completion
# Shellcode (only necessary if global completion is not activated - see Global completion below), to be put in e.g. .bashrc:
# eval "$(register-python-argcomplete v2e.py)"
argcomplete.autocomplete(parser)
args = parser.parse_args()

if __name__ == "__main__":
    overwrite = args.overwrite
    output_folder: str = args.output_folder
    f = not overwrite and Path(output_folder).exists() and os.listdir(output_folder)
    if f:
        logger.error(
            "output folder %s already exists\n it holds files %s\n - use --overwrite",
            Path(output_folder).resolve(),
            f,
        )
        quit()

    if not Path(output_folder).exists():
        logger.info("making output folder %s", output_folder)
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        # os.mkdir(output_folder)

    if (args.output_width is not None) ^ (args.output_width is not None):
        logger.error("provide both or neither of output_width and output_height")
        quit()
    input_file = args.input
    if not input_file:
        input_file = inputDDDFileDialog()
        if not input_file:
            logger.info("no file selected, quitting")
            quit()

    output_width = args.output_width
    output_height = args.output_height
    if (output_width is None) ^ (output_height is None):
        logger.error("set neither or both of output_width and output_height")
        quit()

    exposure_mode, exposure_val, area_dimension = v2e_check_dvs_exposure_args(args)
    if not exposure_mode == ExposureMode.DURATION:
        raise ValueError(
            f"only dvs_exposure=duration is currently supported (mode {exposure_mode} not allowed)"
        )

    write_args_info(args, Path(output_folder))

    dvsFps = 1.0 / exposure_val  # type: ignore
    start_time = args.start_time
    stop_time = args.stop_time
    slowdown_factor = args.slowdown_factor
    pos_thres = args.pos_thres
    neg_thres = args.neg_thres
    sigma_thres = args.sigma_thres
    cutoff_hz = args.cutoff_hz
    leak_rate_hz = args.leak_rate_hz
    shot_noise_rate_hz = args.shot_noise_rate_hz
    dvs_vid: str = args.dvs_vid
    dvs_vid_full_scale: int = args.dvs_vid_full_scale
    dvs_h5 = args.dvs_h5
    # dvs_np = args.dvs_np
    dvs_aedat2 = args.dvs_aedat2
    dvs_text = args.dvs_text
    vid_orig = args.vid_orig
    vid_slomo = args.vid_slomo
    preview = not args.no_preview
    rotate180 = args.rotate180
    numpy_output = args.numpy_output

    if numpy_output:
        allEventsReal = np.empty((0, 4), float)
        allEventsFake = np.empty((0, 4), float)

    memoryLimit = 1e9  # print warnings about excessive memory use in linux, increased by 1GB chunks

    import time

    time_run_started = time.time()

    # input file checking
    if not input_file or not Path(input_file).exists():
        logger.error("input file %s does not exist", input_file)
        quit()

    logger.info("opening output files")
    slomo = SuperSloMo(
        model=args.slomo_model,
        auto_upsample=False,
        upsampling_factor=args.slowdown_factor,
        video_path=output_folder,
        vid_orig=vid_orig,
        vid_slomo=vid_slomo,
        preview=preview,
    )
    dvsVidReal = str(dvs_vid).replace(".avi", "-real.avi")
    dvsVidFake = str(dvs_vid).replace(".avi", "-fake.avi")
    emulator = EventEmulator(
        pos_thres=pos_thres,
        neg_thres=neg_thres,
        sigma_thres=sigma_thres,
        cutoff_hz=cutoff_hz,
        leak_rate_hz=leak_rate_hz,
        shot_noise_rate_hz=shot_noise_rate_hz,
        output_folder=output_folder,
        dvs_h5=dvs_h5,
        dvs_aedat2=dvs_aedat2,
        dvs_text=dvs_text,
    )
    eventRendererReal = EventRenderer(
        exposure_mode=exposure_mode,
        exposure_value=exposure_val,  # type: ignore
        area_dimension=area_dimension,
        output_path=output_folder,
        dvs_vid=dvsVidReal,
        preview=preview,
        full_scale_count=dvs_vid_full_scale,
    )
    eventRendererFake = EventRenderer(
        exposure_mode=exposure_mode,
        exposure_value=exposure_val,  # type: ignore
        area_dimension=area_dimension,
        output_path=output_folder,
        dvs_vid=dvsVidFake,
        preview=preview,
        full_scale_count=dvs_vid_full_scale,
    )
    realDvsAeDatOutput = None

    davisData = DDD20SimpleReader(input_file, rotate180=rotate180)  # type: ignore

    startPacket = (
        davisData.search(timeS=start_time)  # type: ignore
        if start_time
        else davisData.firstPacketNumber
    )
    if startPacket is None:
        raise ValueError(
            "cannot find relative start time " + str(start_time) + "s within recording"
        )
    stopPacket = (
        davisData.search(timeS=stop_time) if stop_time else davisData.numPackets - 1  # type: ignore
    )
    if stopPacket is None:
        raise ValueError(
            "cannot find relative stop time " + str(start_time) + "s within recording"
        )
    if not start_time:
        start_time = 0
    if not stop_time:
        stop_time = davisData.durationS

    srcDurationToBeProcessed = stop_time - start_time
    dvsNumFrames = int(math.floor(dvsFps * srcDurationToBeProcessed))
    if dvsNumFrames == 0:
        dvsNumFrames = 1  # we need at least 1
    dvsDuration = srcDurationToBeProcessed
    dvsPlaybackDuration = dvsNumFrames / args.avi_frame_rate
    dvsFrameTimestamps = np.linspace(
        davisData.firstTimeS + start_time,
        davisData.firstTimeS + start_time + srcDurationToBeProcessed,
        dvsNumFrames,
    )

    logger.info("iterating over input file contents")
    num_frames = 0
    numDvsEvents = 0
    # numOnEvents=0
    # numOffEvents=0
    frames = None
    frame0 = None
    frame1 = None
    dvsFrameTime = 0
    savedEvents = np.empty(
        [0, 4], dtype=float
    )  # to hold extra events in partial last DVS frame
    for i in tqdm(range(startPacket, stopPacket), desc="v2e-ddd20", unit="packet"):
        packet = davisData.readPacket(i)  # type: ignore
        if not packet:
            continue  # empty or could not parse this one
        if stop_time > 0 and packet["timestamp"] > davisData.firstTimeS + stop_time:
            logger.info("\n reached stop time %s", stop_time)
            break
        if packet["etype"] == ddd_h5_reader.DDD20SimpleReader.ETYPE_DVS:
            numDvsEvents += packet["enumber"]
            events = np.array(
                packet["data"], dtype=float
            )  # get just events [:,[ts,x,y,pol]]
            events[:, 0] = events[:, 0] * 1e-6  # us timestamps
            if numpy_output:
                allEventsReal = np.concatenate((allEventsReal, events))
            if not realDvsAeDatOutput and dvs_aedat2:
                filepath = checkAddSuffix(
                    Path(output_folder) / dvs_aedat2, ".aedat"
                ).replace(".aedat", "-real.aedat")
                realDvsAeDatOutput = AEDat2Output(filepath)
            if realDvsAeDatOutput:
                realDvsAeDatOutput.appendEvents(events)
            # prepend saved events if there are some
            events = np.vstack((savedEvents, events))
            # find dvs starting frame index
            ts0 = events[0, 0]
            ts1 = events[-1, 0]
            dt = ts1 - ts0
            dvsFrameStartIdx = np.searchsorted(
                dvsFrameTimestamps, ts0, side="left"
            )  # find first DVS frame
            dvsFrameEndIdx = (
                np.searchsorted(dvsFrameTimestamps, ts1, side="right") - 1
            )  # and last one, we go back -1 more to make sure that the last DVS frame is not partially filled by this packet
            if dvsFrameEndIdx == len(
                dvsFrameTimestamps
            ):  # if ts is past last frame, set to last
                dvsFrameEndIdx -= 1
            endEventIdx = np.searchsorted(
                events[:, 0], dvsFrameTimestamps[dvsFrameEndIdx], side="right"
            )  # find last event that fits into the last DVS frame
            savedEvents = np.copy(
                events[endEventIdx:, :]
            )  # save all events after this for next batch of DVS frames
            theseEvents = events[: endEventIdx - 1, :]
            # this packet spans some time, and we need to render into DVS frames with regular spacing.
            # But render throws all leftover events into the last DVS
            eventRendererReal.render_events_to_frames(
                event_arr=theseEvents, height=output_height, width=output_width
            )

        elif packet["etype"] == ddd_h5_reader.DDD20SimpleReader.ETYPE_APS:
            num_frames += 1
            tmpFrame = frame0
            frame0 = frame1
            frame1 = packet
            if frame0 is not None and frame1 is not None:
                with TemporaryDirectory() as interpFramesFolder:
                    im0 = (frame0["data"] / 256).astype(np.uint8)
                    im1 = (frame1["data"] / 256).astype(np.uint8)
                    # im1=frame1['data'].astype(np.uint8)
                    twoFrames = np.stack([im0, im1], axis=0)
                    slomo.interpolate(
                        twoFrames, interpFramesFolder
                    )  # interpolated frames are stored to tmpfolder as 1.png, 2.png, etc
                    interpFramesFilenames = all_images(
                        interpFramesFolder
                    )  # read back to memory
                    n = len(interpFramesFilenames)  # number of interpolated frames
                    events = np.empty((0, 4), float)
                    # Interpolating the 2 frames f0 to f1 results in n frames f0 fi0 fi1 ... fin-2 f1
                    # The endpoint frames are same as input.
                    # If we pass these to emulator repeatedly,
                    # then the f1 frame from past loop is the same as the f0 frame in the next iteration.
                    # For emulation, we should pass in to the emulator only up to the last interpolated frame,
                    # since the next iteration will pass in the f1 from previous iteration.

                    # compute times of output integrated frames
                    interpTimes = np.linspace(
                        start=frame0["timestamp"],
                        stop=frame1["timestamp"],
                        num=n,
                        endpoint=True,
                    )
                    for i in range(
                        n - 1
                    ):  # for each interpolated frame up to last; use n-1 because we get last interpolated frame as first frame next time
                        fr = read_image(interpFramesFilenames[i])
                        newEvents = emulator.generate_events(fr, interpTimes[i])
                        if newEvents is not None:
                            events = np.append(events, newEvents, axis=0)
                    events = np.array(events)  # remove first None element
                    if numpy_output:
                        allEventsFake = np.concatenate((allEventsFake, events))
                    eventRendererFake.render_events_to_frames(
                        events, height=output_height, width=output_width
                    )
        if numpy_output and platform.startswith("linux"):
            usageRSS_kB = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if usageRSS_kB > memoryLimit / 1024:
                logger.warning(
                    "\n*** memory usage (RSS) is %sB", EngNumber(1024 * usageRSS_kB)
                )
                memoryLimit += 1e9

    if output_folder and numpy_output:
        np.save(Path(output_folder) / "dvs_real.npy", allEventsReal)
        np.save(Path(output_folder) / "dvs_v2e.npy", allEventsFake)
        logger.info("saved numpy files with real and v2e events to %s", output_folder)

    logger.info("done; see output folder " + str(args.output_folder))
    totalTime = time.time() - time_run_started
    framePerS = num_frames / totalTime
    sPerFrame = 1 / framePerS
    throughputStr = (
        (str(EngNumber(framePerS)) + "fr/s")
        if framePerS > 1
        else (str(EngNumber(sPerFrame)) + "s/fr")
    )
    logger.info(
        "done processing %s frames in %ss (%s)\n see output folder %s",
        num_frames,
        EngNumber(totalTime),
        throughputStr,
        output_folder,
    )
    logger.info(
        "generated total %s events (%s on, %s off)",
        EngNumber(emulator.num_events_total),
        EngNumber(emulator.num_events_on),
        EngNumber(emulator.num_events_off),
    )
    logger.info(
        "avg event rate %sHz (%sHz on, %sHz off)",
        EngNumber(emulator.num_events_total / srcDurationToBeProcessed),
        EngNumber(emulator.num_events_on / srcDurationToBeProcessed),
        EngNumber(emulator.num_events_off / srcDurationToBeProcessed),
    )
    try:
        desktop.open(Path(output_folder).resolve())  # type: ignore
    except Exception as e:
        logger.warning("%s: could not open %s in desktop", e, output_folder)
    eventRendererFake.cleanup()
    eventRendererReal.cleanup()
    slomo.cleanup()
    v2e_quit()

import os
import re

with open("v2ecore/renderer.py", "r") as f:
    text = f.read()

text = text.replace(
    "import numpy as np\n",
    "import numpy as np\nfrom typing import Any, Dict, List, Optional, Tuple, Union, cast\n",
)
text = text.replace(
    "from tqdm import tqdm\n", "from tqdm import tqdm  # type: ignore\n"
)
text = text.replace(
    "def njit(*args, **kwargs):", "def njit(*args: Any, **kwargs: Any) -> Any:"
)
text = text.replace("def decorator(func):", "def decorator(func: Any) -> Any:")
text = text.replace(
    """    def __init__(
            self,
            full_scale_count=3,
            output_path=None,
            dvs_vid=None,
            preview=False,
            exposure_mode=ExposureMode.DURATION,  # 'count', 'area-count'
            exposure_value=1 / 300.0,
            area_dimension=None,
            # suffix using dvs_vid file name for the frame times
            # when not using constant_time
            frame_times_suffix='-frame_times.txt',
            avi_frame_rate=30):""",
    """    def __init__(
            self,
            full_scale_count: int = 3,
            output_path: Optional[str] = None,
            dvs_vid: Optional[str] = None,
            preview: bool = False,
            exposure_mode: ExposureMode = ExposureMode.DURATION,  # 'count', 'area-count'
            exposure_value: float = 1 / 300.0,
            area_dimension: Optional[int] = None,
            # suffix using dvs_vid file name for the frame times
            # when not using constant_time
            frame_times_suffix: str = '-frame_times.txt',
            avi_frame_rate: int = 30) -> None:""",
)

text = text.replace("self.width = None", "self.width: Optional[int] = None")
text = text.replace("self.height = None", "self.height: Optional[int] = None")
text = text.replace(
    "self.full_scale_count = full_scale_count",
    "self.full_scale_count: int = full_scale_count",
)
text = text.replace("self.accum_mode = 'duration'", "self.accum_mode: str = 'duration'")
text = text.replace(
    "self.dvs_frame_times_suffix = frame_times_suffix",
    "self.dvs_frame_times_suffix: str = frame_times_suffix",
)
text = text.replace(
    "self.frame_rate_hz = None", "self.frame_rate_hz: Optional[float] = None"
)
text = text.replace("self.event_count = None", "self.event_count: Optional[int] = None")
text = text.replace(
    "self.frameIntevalS = None", "self.frameIntevalS: Optional[float] = None"
)
text = text.replace(
    "self.avi_frame_rate = avi_frame_rate", "self.avi_frame_rate: int = avi_frame_rate"
)
text = text.replace(
    "self.area_counts = None  # 2d array of counts",
    "self.area_counts: Optional[Any] = None  # 2d array of counts",
)
text = text.replace("self.area_count = None", "self.area_count: Optional[int] = None")
text = text.replace(
    "self.area_dimension = area_dimension",
    "self.area_dimension: Optional[int] = area_dimension",
)
text = text.replace(
    "self.video_output_file_name = dvs_vid",
    "self.video_output_file_name: Optional[str] = dvs_vid",
)
text = text.replace(
    "self.video_output_file = None", "self.video_output_file: Optional[Any] = None"
)
text = text.replace(
    "self.frame_times_output_file = None",
    "self.frame_times_output_file: Optional[Any] = None",
)

text = text.replace("raise (f'exposure mode", "raise ValueError(f'exposure mode")

text = text.replace("def cleanup(self):", "def cleanup(self) -> None:")
text = text.replace(
    "def _check_outputs_open(self):", "def _check_outputs_open(self) -> None:"
)

text = text.replace(
    """            self.video_output_file = video_writer(
                fn, self.height, self.width,
                frame_rate=self.avi_frame_rate)""",
    """            self.video_output_file = video_writer(  # type: ignore
                fn, self.height, self.width,
                frame_rate=self.avi_frame_rate)""",
)

text = text.replace(
    """    def render_events_to_frames(self, event_arr: np.ndarray,
                                height: int, width: int,
                                return_frames=False) -> np.ndarray:""",
    """    def render_events_to_frames(self, event_arr: Optional[Any],
                                height: int, width: int,
                                return_frames: bool = False) -> Optional[Any]:""",
)

text = text.replace(
    """        if self.exposure_mode == ExposureMode.DURATION:
            if self.currentFrameStartTime is None:
                self.currentFrameStartTime = ts[0]  # initialize this frame

            nextFrameStartTs = self.currentFrameStartTime + self.frameIntevalS""",
    """        if self.exposure_mode == ExposureMode.DURATION:
            if self.currentFrameStartTime is None:
                self.currentFrameStartTime = ts[0]  # initialize this frame
            assert self.currentFrameStartTime is not None
            assert self.frameIntevalS is not None
            nextFrameStartTs = self.currentFrameStartTime + self.frameIntevalS""",
)

text = text.replace(
    """        if self.exposure_mode == ExposureMode.AREA_COUNT and \\
                self.area_counts is None:
            nw = 1 + self.width // self.area_dimension
            nh = 1 + self.height // self.area_dimension
            self.area_counts = np.zeros(shape=(nw, nh), dtype=int)

        returnedFrames = None""",
    """        if self.exposure_mode == ExposureMode.AREA_COUNT and \\
                self.area_counts is None:
            assert self.width is not None and self.area_dimension is not None
            assert self.height is not None
            nw = 1 + self.width // self.area_dimension
            nh = 1 + self.height // self.area_dimension
            self.area_counts = np.zeros(shape=(nw, nh), dtype=int)

        returnedFrames: Optional[Any] = None""",
)

text = text.replace(
    """        @jit(nopython=True)
        def search_duration_idx(ts, curr_start, next_start):
            start = np.searchsorted(ts, curr_start, side="left")
            end = np.searchsorted(ts, next_start, side="right")
            return start, end""",
    """        @jit(nopython=True)  # type: ignore
        def search_duration_idx(ts: Any, curr_start: float, next_start: float) -> Tuple[int, int]:
            start = np.searchsorted(ts, curr_start, side="left")
            end = np.searchsorted(ts, next_start, side="right")
            return int(start), int(end)""",
)

text = text.replace(
    """        @jit(nopython=True)
        def normalize_frame(curr_frame, full_scale_count):
            return (curr_frame + full_scale_count) / float(
                full_scale_count * 2)""",
    """        @jit(nopython=True)  # type: ignore
        def normalize_frame(curr_frame: Any, full_scale_count: int) -> Any:
            return (curr_frame + full_scale_count) / float(
                full_scale_count * 2)""",
)

text = text.replace(
    """        @jit(nopython=True)
        def compute_area_counts(events, area_counts,
                                area_count, area_dimension, start):
            #  new_area_counts = np.copy(area_counts)
            ev_idx = start
            for ev_idx in range(start, events.shape[0]):
                x = int(events[ev_idx, 1] // area_dimension)
                y = int(events[ev_idx, 2] // area_dimension)
                count = 1 + area_counts[x, y]
                area_counts[x, y] = count
                if count >= area_count:
                    area_counts = np.zeros_like(area_counts)
                    break

            return area_counts, ev_idx""",
    """        @jit(nopython=True)  # type: ignore
        def compute_area_counts(events: Any, area_counts: Any,
                                area_count: int, area_dimension: int, start: int) -> Tuple[Any, int]:
            #  new_area_counts = np.copy(area_counts)
            ev_idx = start
            for ev_idx in range(start, events.shape[0]):
                x = int(events[ev_idx, 1] // area_dimension)
                y = int(events[ev_idx, 2] // area_dimension)
                count = 1 + area_counts[x, y]
                area_counts[x, y] = count
                if count >= area_count:
                    area_counts = np.zeros_like(area_counts)
                    break

            return area_counts, ev_idx""",
)

text = text.replace(
    """            elif self.exposure_mode == ExposureMode.COUNT:
                start = thisFrameIdx
                end = start + self.event_count""",
    """            elif self.exposure_mode == ExposureMode.COUNT:
                assert self.event_count is not None
                start = thisFrameIdx
                end = start + self.event_count""",
)

text = text.replace(
    """                if self.exposure_mode == ExposureMode.DURATION:
                    # increase time to next frame
                    self.currentFrameStartTime += self.frameIntevalS
                    nextFrameStartTs = self.currentFrameStartTime + \\
                                       self.frameIntevalS""",
    """                if self.exposure_mode == ExposureMode.DURATION:
                    assert self.currentFrameStartTime is not None
                    assert self.frameIntevalS is not None
                    # increase time to next frame
                    self.currentFrameStartTime += self.frameIntevalS
                    nextFrameStartTs = self.currentFrameStartTime + \\
                                       self.frameIntevalS""",
)

text = text.replace(
    """                        exposure_mode_cond = (
                                self.exposure_mode == ExposureMode.COUNT or
                                self.exposure_mode == ExposureMode.AREA_COUNT)
                        t = (ts[start] + ts[end]) / 2 if exposure_mode_cond else \\
                            self.currentFrameStartTime + self.frameIntevalS / 2""",
    """                        exposure_mode_cond = (
                                self.exposure_mode == ExposureMode.COUNT or
                                self.exposure_mode == ExposureMode.AREA_COUNT)
                        if exposure_mode_cond:
                            t = (ts[start] + ts[end]) / 2
                        else:
                            assert self.currentFrameStartTime is not None
                            assert self.frameIntevalS is not None
                            t = self.currentFrameStartTime + self.frameIntevalS / 2""",
)

text = text.replace(
    """                        v2e_quit()""",
    """                        v2e_quit()  # type: ignore""",
)

text = text.replace(
    """    def accumulate_event_frame(self, events, histrange):""",
    """    def accumulate_event_frame(self, events: Any, histrange: Any) -> None:""",
)

text = text.replace(
    """            if self.exposure_mode == ExposureMode.DURATION:
                # find first event that is after the current frames start time""",
    """            if self.exposure_mode.value == ExposureMode.DURATION.value:
                # find first event that is after the current frames start time""",
)

text = text.replace(
    """            elif self.exposure_mode == ExposureMode.COUNT:
                assert self.event_count is not None""",
    """            elif self.exposure_mode.value == ExposureMode.COUNT.value:
                assert self.event_count is not None""",
)

text = text.replace(
    """            elif self.exposure_mode == ExposureMode.AREA_COUNT:
                start = thisFrameIdx""",
    """            elif self.exposure_mode.value == ExposureMode.AREA_COUNT.value:
                start = thisFrameIdx""",
)

text = text.replace(
    """            elif self.exposure_mode == ExposureMode.SOURCE:
                start = 0""",
    """            elif self.exposure_mode.value == ExposureMode.SOURCE.value:
                start = 0""",
)

text = text.replace(
    """            if not doneWithTheseEvents or self.exposure_mode==ExposureMode.SOURCE:""",
    """            if not doneWithTheseEvents or self.exposure_mode.value==ExposureMode.SOURCE.value:""",
)

text = text.replace(
    """                if self.exposure_mode == ExposureMode.DURATION:
                    assert self.currentFrameStartTime is not None""",
    """                if self.exposure_mode.value == ExposureMode.DURATION.value:
                    assert self.currentFrameStartTime is not None""",
)

text = text.replace(
    """                elif self.exposure_mode == ExposureMode.COUNT or \\
                        self.exposure_mode == ExposureMode.AREA_COUNT:
                    thisFrameIdx = end""",
    """                elif self.exposure_mode.value == ExposureMode.COUNT.value or \\
                        self.exposure_mode.value == ExposureMode.AREA_COUNT.value:
                    thisFrameIdx = end""",
)

text = text.replace(
    """                elif self.exposure_mode==ExposureMode.SOURCE:
                    pass""",
    """                elif self.exposure_mode.value==ExposureMode.SOURCE.value:
                    pass""",
)

text = text.replace(
    """                    if self.exposure_mode==ExposureMode.SOURCE:
                        t=ts[0] if len(ts)>0 else float('nan')""",
    """                    if self.exposure_mode.value==ExposureMode.SOURCE.value:
                        t=ts[0] if len(ts)>0 else float('nan')""",
)

text = text.replace(
    """                        exposure_mode_cond = (
                                self.exposure_mode == ExposureMode.COUNT or
                                self.exposure_mode == ExposureMode.AREA_COUNT)""",
    """                        exposure_mode_cond = (
                                self.exposure_mode.value == ExposureMode.COUNT.value or
                                self.exposure_mode.value == ExposureMode.AREA_COUNT.value)""",
)

with open("v2ecore/renderer.py", "w") as f:
    f.write(text)

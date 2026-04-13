import numpy as np

from v2ecore.renderer import EventRenderer
from v2ecore.renderer import ExposureMode


def test_event_renderer_duration(tmp_path):  # type: ignore
    output_path = str(tmp_path)
    # We need enough events so that the first frame finishes
    # and there's a subsequent event to trigger frame emission.
    events = np.array(
        [
            [0.01, 10, 10, 1],
            [0.02, 20, 20, 0],
            [
                0.03,
                30,
                30,
                1,
            ],  # > 0.025 (next frame boundary), triggers emission of frame 1
            [
                0.04,
                40,
                40,
                0,
            ],  # extra event so `end < numEvents - 1` is true when processing frame 1
        ],
        dtype=np.float64,
    )

    renderer = EventRenderer(
        output_path=output_path,
        dvs_vid="dvs_duration.avi",
        exposure_mode=ExposureMode.DURATION,
        exposure_value=0.015,  # 15ms duration per frame
    )

    frames = renderer.render_events_to_frames(
        events, height=260, width=346, return_frames=True
    )

    renderer.cleanup()

    assert frames is not None
    assert frames.shape[0] >= 1
    assert frames.shape[1] == 260
    assert frames.shape[2] == 346


def test_event_renderer_count(tmp_path):  # type: ignore
    output_path = str(tmp_path)
    events = np.array(
        [[0.01, 10, 10, 1], [0.02, 20, 20, 0], [0.03, 30, 30, 1]], dtype=np.float64
    )

    renderer = EventRenderer(
        output_path=output_path,
        dvs_vid="dvs_count.avi",
        exposure_mode=ExposureMode.COUNT,
        exposure_value=1,  # 1 event per frame
    )

    frames = renderer.render_events_to_frames(
        events, height=260, width=346, return_frames=True
    )

    renderer.cleanup()

    assert frames is not None
    assert frames.shape[0] >= 1
    assert frames.shape[1] == 260
    assert frames.shape[2] == 346

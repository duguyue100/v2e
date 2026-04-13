import numpy as np

from v2ecore.emulator import EventEmulator


def test_emulator_basic():  # type: ignore
    # 346x260 random frames
    w, h = 346, 260
    emulator = EventEmulator(
        pos_thres=0.2, neg_thres=0.2, output_width=w, output_height=h, device="cpu"
    )

    # 3 frames of random noise
    frames = [np.random.rand(h, w) * 255 for _ in range(3)]
    times = [0.0, 0.01, 0.02]

    events_generated = False
    for i, frame in enumerate(frames):
        events = emulator.generate_events(frame, times[i])

        if events is not None:
            events_generated = True
            assert events.ndim == 2
            assert events.shape[1] == 4
            # Event format: [t, x, y, p]
            # Verify basic bounds
            assert np.all(events[:, 0] >= 0.0)
            assert np.all((events[:, 1] >= 0) & (events[:, 1] < w))
            assert np.all((events[:, 2] >= 0) & (events[:, 2] < h))
            assert np.all(np.isin(events[:, 3], [-1, 1]))
    assert events_generated, "Expected some events to be generated"


def test_emulator_sci_mode():  # type: ignore
    w, h = 346, 260
    emulator = EventEmulator(
        pos_thres=0.2,
        neg_thres=0.2,
        output_width=w,
        output_height=h,
        device="cpu",
        scidvs=True,
    )

    frames = [np.random.rand(h, w) * 255 for _ in range(3)]
    times = [0.0, 0.01, 0.02]

    events_generated = False
    for i, frame in enumerate(frames):
        events = emulator.generate_events(frame, times[i])
        if events is not None:
            events_generated = True
            assert events.ndim == 2
            assert events.shape[1] == 4
    assert events_generated, "Expected some events to be generated"


def test_emulator_cs_mode():  # type: ignore
    w, h = 346, 260
    emulator = EventEmulator(
        pos_thres=0.2,
        neg_thres=0.2,
        output_width=w,
        output_height=h,
        device="cpu",
        cs_lambda_pixels=2.0,
        cs_tau_p_ms=10.0,
    )

    frames = [np.random.rand(h, w) * 255 for _ in range(3)]
    times = [0.0, 0.01, 0.02]

    events_generated = False
    for i, frame in enumerate(frames):
        events = emulator.generate_events(frame, times[i])
        if events is not None:
            events_generated = True
            assert events.ndim == 2
            assert events.shape[1] == 4
    assert events_generated, "Expected some events to be generated"

import torch
import numpy as np
import pytest

from v2ecore.emulator_utils import (
    lin_log,
    low_pass_filter,
    subtract_leak_current,
    compute_event_map,
    generate_shot_noise,
)


def test_lin_log():
    # Input tensor
    x = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
    threshold = 20.0

    y = lin_log(x, threshold=threshold)

    # Expected calculations
    # threshold = 20 -> f = (1/20)*ln(20) = 0.1497866
    # x = 10 -> 10 * 0.1497866 = 1.497866
    # x = 20 -> ln(20) = 2.995732
    # x = 30 -> ln(30) = 3.401197
    expected = torch.tensor(
        [10.0 * (1.0 / 20.0) * np.log(20.0), np.log(20.0), np.log(30.0)],
        dtype=torch.float32,
    )

    # Rounded to 1e8 internally in the function
    rounding = 1e8
    expected = torch.round(expected * rounding) / rounding

    torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-5)


def test_low_pass_filter():
    log_new_frame = torch.tensor([1.0, 2.0])
    lp_log_frame = torch.tensor([0.0, 1.0])
    delta_time = 0.1
    cutoff_hz = 10.0
    inten01 = torch.tensor([0.5, 1.0])

    # If cutoff <= 0, return log_new_frame
    res_no_cutoff = low_pass_filter(log_new_frame, lp_log_frame, inten01, delta_time, 0)
    torch.testing.assert_close(res_no_cutoff, log_new_frame)

    # If cutoff > 0
    res = low_pass_filter(log_new_frame, lp_log_frame, inten01, delta_time, cutoff_hz)

    tau = 1.0 / (np.pi * 2 * cutoff_hz)
    eps = inten01 * (delta_time / tau)
    eps = torch.clamp(eps, max=1.0)

    expected = (1.0 - eps) * lp_log_frame + eps * log_new_frame
    torch.testing.assert_close(res, expected)


def test_subtract_leak_current():
    base_log_frame = torch.tensor([1.0, 2.0, 3.0])
    leak_rate_hz = 0.1
    delta_time = 0.5
    pos_thres = torch.tensor([0.2, 0.2, 0.2])
    leak_jitter_fraction = 0.0  # Deterministic
    noise_rate_array = torch.tensor([1.0, 0.5, 0.0])

    res = subtract_leak_current(
        base_log_frame,
        leak_rate_hz,
        delta_time,
        pos_thres,
        leak_jitter_fraction,
        noise_rate_array,
    )

    curr_leak_rate = leak_rate_hz * noise_rate_array
    delta_leak = delta_time * curr_leak_rate * pos_thres
    expected = base_log_frame - delta_leak

    torch.testing.assert_close(res, expected)


def test_compute_event_map():
    diff_frame = torch.tensor([0.5, 1.5, -0.8, -2.5, 0.0])
    pos_thres = torch.tensor(1.0)
    neg_thres = torch.tensor(1.0)

    pos_evts, neg_evts = compute_event_map(diff_frame, pos_thres, neg_thres)

    expected_pos = torch.tensor([0, 1, 0, 0, 0], dtype=torch.int32)
    expected_neg = torch.tensor([0, 0, 0, 2, 0], dtype=torch.int32)

    torch.testing.assert_close(pos_evts, expected_pos)
    torch.testing.assert_close(neg_evts, expected_neg)


def test_generate_shot_noise():
    shot_noise_rate_hz = 0.0  # 0 rate -> no noise
    delta_time = 0.1
    shot_noise_inten_factor = 1.0
    inten01 = torch.tensor([0.5, 0.8])
    pos_thres_pre_prob = torch.tensor([1.0, 1.0])
    neg_thres_pre_prob = torch.tensor([1.0, 1.0])

    shot_on, shot_off = generate_shot_noise(
        shot_noise_rate_hz,
        delta_time,
        shot_noise_inten_factor,
        inten01,
        pos_thres_pre_prob,
        neg_thres_pre_prob,
    )

    # 0 rate -> probabilities are 0 -> all false
    expected_on = torch.tensor([False, False], dtype=torch.bool)
    expected_off = torch.tensor([False, False], dtype=torch.bool)

    torch.testing.assert_close(shot_on, expected_on)
    torch.testing.assert_close(shot_off, expected_off)

    # High rate -> all noise
    shot_noise_rate_hz = 1000000.0
    shot_on_high, shot_off_high = generate_shot_noise(
        shot_noise_rate_hz,
        delta_time,
        shot_noise_inten_factor,
        inten01,
        pos_thres_pre_prob,
        neg_thres_pre_prob,
    )

    expected_on_high = torch.tensor([True, True], dtype=torch.bool)
    expected_off_high = torch.tensor([True, True], dtype=torch.bool)

    torch.testing.assert_close(shot_on_high, expected_on_high)
    torch.testing.assert_close(shot_off_high, expected_off_high)
